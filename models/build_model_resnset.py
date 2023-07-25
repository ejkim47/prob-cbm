
from email.mime import base
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from torchvision.models import resnet18

from utils.util import weights_init, ent_loss_fn
from .module import PIENet, UncertaintyModuleImage, MC_dropout
from utils.loss import sample_gaussian_tensors, batchwise_cdist


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask)

class ConceptConvModelBase(nn.Module):
    def __init__(self, backbone=resnet18, pretrained=True, train_class_mode='sequential'):
        super(ConceptConvModelBase, self).__init__()
        self.train_class_mode = train_class_mode
        self.use_dropout = False

        base_model = backbone(pretrained=pretrained, num_classes=1000)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.avgpool = base_model.avgpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.d_model = self.layer4[-1].conv2.out_channels

        self.cnn_module = nn.ModuleList([self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4])

    def forward_basic(self, x, avgpool=True, sample=False):
        if hasattr(self, 'features'):
            x = self.features(x)
            return x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer2(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer3(x)
        if self.use_dropout:
            x = MC_dropout(x, p=0.2, mask=sample)
        x = self.layer4(x)

        if avgpool:
            return self.avgpool(x)
        return x


class ProbConceptModel(ConceptConvModelBase):
    def __init__(self, num_concepts, backbone=resnet18, pretrained=True, num_classes=200, hidden_dim=128, n_samples_inference=7, \
        use_neg_concept=False, pred_class=False, use_scale=False, \
        activation_concept2class='prob', token2concept=None, train_class_mode='sequential', **kwargs):
        super(ProbConceptModel, self).__init__(backbone=backbone, pretrained=pretrained)

        self.group2concept = token2concept
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.activation_concept2class = activation_concept2class
        self.train_class_mode = train_class_mode
        self.use_scale = use_scale

        self.mean_head = nn.Sequential(nn.Conv2d(self.d_model, num_concepts * hidden_dim, kernel_size=1), nn.ReLU(), \
            nn.Conv2d(num_concepts * hidden_dim, num_concepts * hidden_dim, kernel_size=1, groups=num_concepts), nn.ReLU(),
            nn.Conv2d(num_concepts * hidden_dim, num_concepts * hidden_dim, kernel_size=1, groups=num_concepts))
        self.logsigma_head = nn.Sequential(nn.Conv2d(self.d_model, num_concepts * hidden_dim, kernel_size=1), nn.LeakyReLU(), \
            nn.Conv2d(num_concepts * hidden_dim, num_concepts * hidden_dim, kernel_size=1, groups=num_concepts), nn.LeakyReLU(),
            nn.Conv2d(num_concepts * hidden_dim, num_concepts * hidden_dim, kernel_size=1, groups=num_concepts))
        
        weights_init(self.mean_head)
        weights_init(self.logsigma_head)

        self.use_neg_concept = use_neg_concept
        n_neg_concept = 1 if use_neg_concept else 0
        self.concept_vectors = nn.Parameter(torch.randn(n_neg_concept+1, num_concepts, hidden_dim), requires_grad=True)
        negative_scale = kwargs.get('init_negative_scale', 1) * torch.ones(1)
        shift = kwargs.get('init_shift', 0) * torch.ones(1)
        nn.init.trunc_normal_(self.concept_vectors, std=1.0 / math.sqrt(hidden_dim))

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.n_samples_inference = n_samples_inference

        self.pred_class = pred_class
        if pred_class:
            self.head = nn.Linear(num_concepts, num_classes)
            if use_scale:
                scale = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
                self.register_parameter('scale', scale)
            weights_init(self.head)

    def match_prob(self, sampled_image_features, sampled_attr_features, negative_scale=None, shift=None, reduction='mean'):
        negative_scale = self.negative_scale if negative_scale is None else negative_scale
        shift = self.shift if shift is None else shift
        if not self.use_neg_concept:
            sampled_attr_features = sampled_attr_features[1:] if sampled_attr_features.shape[0] > 1 else sampled_attr_features
            distance = batchwise_cdist(sampled_image_features, sampled_attr_features)

            distance = distance.float()
            logits = -negative_scale.view(1, -1, 1) * distance + shift.view(1, -1, 1)
            prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

            if reduction == 'none':
                return logits, prob
            else:
                return logits.mean(axis=-1), prob.mean(axis=-1)
        else:
            distance = batchwise_cdist(sampled_image_features, sampled_attr_features)
            distance = distance.permute(0, 2, 3, 1)

            logits = -self.negative_scale.view(1, -1, 1, 1) * distance
            prob = torch.nn.functional.softmax(logits, dim=-1)
            if reduction == 'none':
                return logits, prob
            return logits.mean(axis=-2), prob.mean(axis=-2)

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty

    def get_class_uncertainty(self, pred_concept_logsigma, weight):
        all_logsigma = pred_concept_logsigma.view(pred_concept_logsigma.shape[0], -1)
        cov = all_logsigma
        cov = torch.eye(cov.shape[1]).unsqueeze(0).to(cov.device) * cov.unsqueeze(-1)
        full_cov = F.linear((F.linear(cov, weight)).transpose(1, 2), weight)
        _, s, _ = torch.linalg.svd(full_cov)
        C = (s + 1e-10).log()
        return C.mean(dim=1).exp()
    
    def get_uncertainty_with_matching_prob(self, sampled_embeddings, negative_scale=None): # * x n_samples x dim
        self_distance = torch.sqrt(((sampled_embeddings.unsqueeze(-2) - sampled_embeddings.unsqueeze(-3)) ** 2).mean(-1) + 1e-10)
        eye = 1 - torch.eye(self_distance.size(-2)).view(-1)
        eye = eye.nonzero().contiguous().view(-1)
        logits = -self_distance.view(*sampled_embeddings.shape[:-2], -1)[..., eye] * (negative_scale if negative_scale is not None else 1)
        uncertainty = 1 - torch.sigmoid(logits).mean(dim=-1)
        return uncertainty
    
    def sample_embeddings(self, x, n_samples_inference=None):
        n_samples_inference = self.n_samples_inference if n_samples_inference is None else n_samples_inference
        B = x.shape[0]
        feature = self.forward_basic(x)
        pred_concept_mean = self.mean_head(feature).view(B, self.num_concepts, -1)
        pred_concept_logsigma = self.logsigma_head(feature).view(B, self.num_concepts, -1)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)

        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference)

        return {'pred_embeddings': pred_embeddings, 'pred_mean': pred_concept_mean, 'pred_logsigma': pred_concept_logsigma}


    def forward(self, x, **kwargs):
        B = x.shape[0]
        feature = self.forward_basic(x)
        pred_concept_mean = self.mean_head(feature).view(B, self.num_concepts, -1)
        pred_concept_logsigma = self.logsigma_head(feature).view(B, self.num_concepts, -1)
        pred_concept_logsigma = torch.clip(pred_concept_logsigma, max=10)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])
            concept_logsigma = torch.cat([concept_logsigma, concept_logsigma]) if concept_logsigma is not None else None

        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        concept_embeddings = concept_mean.unsqueeze(-2)
        
        concept_logit, concept_prob = self.match_prob(pred_embeddings, concept_embeddings)
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)

        if self.concept_pos_idx.sum() > 1:
            out_concept_prob, out_concept_idx = concept_prob[:, self.concept_pos_idx==1].max(dim=1)
        else:
            out_concept_prob = concept_prob[..., 1] if self.use_neg_concept else concept_prob

        out_dict = {'pred_concept_prob': out_concept_prob, 'pred_concept_uncertainty': concept_uncertainty, 'pred_concept_logit': concept_logit, 'pred_embeddings': pred_embeddings, 'concept_embeddings': concept_embeddings, 'pred_mean': pred_concept_mean, 'pred_logsigma': pred_concept_logsigma, \
            'concept_mean':concept_mean, 'concept_logsigma': concept_logsigma, 'shift': self.shift, 'negative_scale': self.negative_scale, 'pred_embeddings_detach': pred_embeddings_detach, 'concept_pos_idx': self.concept_pos_idx}

        if self.pred_class:
            out_concept_prob_d = out_concept_prob.detach()

            if hasattr(self, 'scale'):
                class_logits = self.head(out_concept_prob_d * self.scale.pow(2))
            else:
                class_logits = self.head(out_concept_prob_d)

            out_dict['pred_class_logit'] = class_logits

        return out_dict, {}
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'concept_vectors', 'class_vectors', 'shift', 'negative_scale'}



class ProbCBM(ProbConceptModel):
    def __init__(self, num_concepts, hidden_dim=128, num_classes=200, class_hidden_dim=None, intervention_prob=False, use_class_emb_from_concept=False, use_probabilsitic_concept=True, **kwargs):
        super(ProbCBM, self).__init__(num_concepts=num_concepts, hidden_dim=hidden_dim, num_classes=num_classes, **kwargs)

        self.intervention_prob = intervention_prob
        self.use_class_emb_from_concept = use_class_emb_from_concept
        self.use_probabilsitic_concept = use_probabilsitic_concept
        del self.mean_head
        del self.logsigma_head
        
        self.mean_head = nn.ModuleList([PIENet(1, self.d_model, hidden_dim, hidden_dim // 2) for _ in range(num_concepts)])
        if self.use_probabilsitic_concept:
            self.logsigma_head = nn.ModuleList([UncertaintyModuleImage(self.d_model, hidden_dim, hidden_dim // 2) for _ in range(num_concepts)])

        if self.pred_class:
            del self.head

            class_hidden_dim = class_hidden_dim if class_hidden_dim is not None else hidden_dim * 7
            if not self.use_class_emb_from_concept:
                self.class_mean = nn.Parameter(torch.randn(num_classes, class_hidden_dim), requires_grad=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * num_concepts, class_hidden_dim)
            )

            if self.use_scale:
                self.class_negative_scale = nn.Parameter(torch.ones(1) * 5, requires_grad=True)

            weights_init(self.head)
        
        del self.mean_head

        self.stem = nn.Sequential(nn.Conv2d(self.d_model, hidden_dim * num_concepts, kernel_size=1),
                                  nn.BatchNorm2d(hidden_dim * num_concepts),
                                  nn.ReLU())
        weights_init(self.stem)
        self.mean_head = nn.ModuleList(
            [PIENet(1, hidden_dim, hidden_dim, hidden_dim) for _ in range(num_concepts)])
        if self.use_probabilsitic_concept:
            del self.logsigma_head
            self.logsigma_head = nn.ModuleList(
                [UncertaintyModuleImage(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_concepts)])

    
    def predict_concept_dist(self, x):
        B = x.shape[0]
        feature = self.forward_basic(x, avgpool=False)
        feature = self.stem(feature)
        feature_avg = self.avgpool(feature).flatten(1)
        feature = feature.view(B, self.num_concepts, -1, feature.shape[-2:].numel()).transpose(2, 3)
        feature_avg = feature_avg.view(B, self.num_concepts, -1)
        pred_concept_mean = torch.stack([mean_head(feature_avg[:, i], feature[:, i])[0] for i, mean_head in enumerate(self.mean_head)], dim=1)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        if self.use_probabilsitic_concept:
            pred_concept_logsigma = torch.stack(
                [logsigma_head(feature_avg[:, i], feature[:, i])['logsigma'] for i, logsigma_head in
                 enumerate(self.logsigma_head)], dim=1)
            pred_concept_logsigma = torch.clip(pred_concept_logsigma, max=10)
            return pred_concept_mean, pred_concept_logsigma
        return pred_concept_mean, None
        
    def forward(self, x, inference_with_sampling=False, n_samples_inference=None, **kwargs):
        B = x.shape[0]
        pred_concept_mean, pred_concept_logsigma = self.predict_concept_dist(x)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])

        if self.use_probabilsitic_concept:
            if self.training:
                pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference)
            else:
                if not inference_with_sampling:
                    pred_embeddings = pred_concept_mean.unsqueeze(2)
                else:
                    n_samples_inference = self.n_samples_inference if n_samples_inference is None else n_samples_inference
                    pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, n_samples_inference)
        else:
            pred_embeddings = pred_concept_mean.unsqueeze(2)

        concept_embeddings = concept_mean.unsqueeze(-2)
        concept_logit, concept_prob = self.match_prob(pred_embeddings, concept_embeddings, reduction='none')

        out_concept_prob = concept_prob[..., 1].mean(dim=-1) if self.use_neg_concept else concept_prob.mean(dim=-1)

        if self.use_probabilsitic_concept:
            concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
            out_dict = {'pred_concept_prob': out_concept_prob, 'pred_concept_uncertainty': concept_uncertainty,
                        'pred_concept_logit': concept_logit, 'pred_embeddings': pred_embeddings,
                        'concept_embeddings': concept_embeddings, 'pred_mean': pred_concept_mean,
                        'pred_logsigma': pred_concept_logsigma, \
                        'concept_mean': concept_mean, 'shift': self.shift,
                        'negative_scale': self.negative_scale}
        else:
            out_dict = {'pred_concept_prob': out_concept_prob,
                        'pred_embeddings': pred_embeddings,
                        'concept_embeddings': concept_embeddings, 'pred_mean': pred_concept_mean,
                        'concept_mean': concept_mean, 'shift': self.shift,
                        'negative_scale': self.negative_scale}

        if self.pred_class:
            if self.train_class_mode in ['sequential', 'independent']:
                pred_embeddings_for_class = pred_embeddings.permute(0, 2, 1, 3).detach()
            elif self.train_class_mode == 'joint':
                pred_embeddings_for_class = pred_embeddings.permute(0, 2, 1, 3)
            else:
                raise NotImplementedError('train_class_mode should be one of [sequential, joint]')
            
            if self.training:
                target_concept_onehot = F.one_hot(kwargs['target_concept'].long(), num_classes=2)
                gt_concept_embedddings = (target_concept_onehot.view(*target_concept_onehot.shape, 1, 1) * concept_embeddings.permute(1, 0, 2, 3).unsqueeze(0)).sum(2)
                if self.train_class_mode == 'independent':
                    pred_embeddings_for_class = sample_gaussian_tensors(gt_concept_embedddings.squeeze(-2), pred_concept_logsigma.detach(), self.n_samples_inference)
                    pred_embeddings_for_class = pred_embeddings_for_class.permute(0, 2, 1, 3).detach()
                if gt_concept_embedddings.shape[2] != self.n_samples_inference:
                    gt_concept_embedddings = gt_concept_embedddings.repeat(1, 1, self.n_samples_inference, 1)
                gt_concept_embedddings = gt_concept_embedddings.permute(0, 2, 1, 3).detach()
                pred_embeddings_for_class = torch.where(torch.rand_like(gt_concept_embedddings[..., :1, :1]) < self.intervention_prob, gt_concept_embedddings, pred_embeddings_for_class)
            
            pred_embeddings_for_class = pred_embeddings_for_class.contiguous().view(*pred_embeddings_for_class.shape[:2], -1)
            pred_embeddings_for_class = self.head(pred_embeddings_for_class)
            out_dict['pred_embeddings_for_class'] = pred_embeddings_for_class
            class_mean = self.class_mean
            out_dict['class_mean'] = class_mean
            distance = torch.sqrt(((pred_embeddings_for_class.unsqueeze(1) - class_mean.unsqueeze(1).repeat(1, self.n_samples_inference if self.training else 1, 1).unsqueeze(0)) ** 2).mean(-1) + 1e-10)
            if self.use_scale:
                distance = self.class_negative_scale * distance
            out_class = F.softmax(-distance, dim=-2)
            out_dict['pred_class_prob'] = out_class.mean(dim=-1)
            if self.use_probabilsitic_concept and not self.training:
                out_dict['pred_class_uncertainty'] = self.get_class_uncertainty(pred_concept_logsigma.exp(), self.head[0].weight)

        return out_dict, {}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'concept_vectors', 'class_vectors', 'shift', 'negative_scale', 'class_mean'}

    @torch.jit.ignore
    def params_to_classify(self):
        output = []
        for n, _ in self.head.named_parameters():
            output.append('head.' + n)
        return output + ['class_mean', 'class_negative_scale']
    
    def predict_concept(self, x):
        pred_concept_mean, pred_concept_logsigma = self.predict_concept_dist(x)
        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)

        if self.use_neg_concept and concept_mean.shape[0] < 2:
            concept_mean = torch.cat([-concept_mean, concept_mean])

        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference)
        concept_embeddings = concept_mean.unsqueeze(-2)

        concept_logit, concept_prob = self.match_prob(pred_embeddings, concept_embeddings, reduction='none')
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)

        out_concept_prob = concept_prob[..., 1].mean(dim=-1) if self.use_neg_concept else concept_prob.mean(dim=-1)

        return {'pred_concept_prob': out_concept_prob, 'pred_concept_uncertainty': concept_uncertainty,
                'pred_embeddings': pred_embeddings, 'pred_logsigma': pred_concept_logsigma}

    def predict_class_with_gt_concepts(self, x, target_concept, order='uncertainty_avg', get_uncertainty=False):
        B = x.shape[0]
        out_dict = self.predict_concept(x)

        concept_mean = F.normalize(self.concept_vectors, p=2, dim=-1)
        concept_embeddings = concept_mean.unsqueeze(-2)
        
        target_concept_onehot = F.one_hot(target_concept.long(), num_classes=2)
        gt_concept_embedddings = (target_concept_onehot.view(*target_concept_onehot.shape, 1, 1) * concept_embeddings.permute(1, 0, 2, 3).unsqueeze(0)).sum(2)

        pred_concept_embeddings = out_dict['pred_embeddings']
        n_groups = self.group2concept.shape[0]
        group2concept = self.group2concept.to(x.device)
        if order == 'uncertainty_avg':
            group_uncertainty = (out_dict['pred_concept_uncertainty'] @ group2concept.t()) / group2concept.sum(dim=1).unsqueeze(0)
            _, intervention_order = group_uncertainty.sort(descending=True, dim=1)
        elif order == 'uncertainty_max':
            group_uncertainty, _ = (out_dict['pred_concept_uncertainty'].unsqueeze(1) * group2concept.unsqueeze(0)).max(dim=-1)
            _, intervention_order = group_uncertainty.sort(descending=True, dim=1)
        else:
            assert isinstance(order, torch.Tensor)
            intervention_order = order.unsqueeze(0).repeat(B, 1)

        all_out_class = []
        all_uncertainty_class = []

        pred_concept_sigma = out_dict['pred_logsigma'].exp()
        for i in range(n_groups + 1):
            if i == 0:
                interventioned_concept_embedddings = pred_concept_embeddings
            else:
                inter_concepts_idx = group2concept[intervention_order[:, :i]].sum(dim=1)
                interventioned_concept_embedddings = torch.where(inter_concepts_idx.view(*inter_concepts_idx.shape, 1, 1) == 1, gt_concept_embedddings, pred_concept_embeddings)
                pred_concept_sigma = torch.where(inter_concepts_idx.view(*inter_concepts_idx.shape, 1) == 1, torch.zeros_like(pred_concept_sigma), pred_concept_sigma)
            interventioned_concept_embedddings = interventioned_concept_embedddings.permute(0, 2, 1, 3).contiguous().view(B, self.n_samples_inference, -1)
            pred_embeddings_for_class = self.head(interventioned_concept_embedddings)
            distance = torch.sqrt(((pred_embeddings_for_class.unsqueeze(1) - self.class_mean.unsqueeze(1).repeat(1, self.n_samples_inference, 1).unsqueeze(0)) ** 2).mean(-1) + 1e-10)
            out_class = F.softmax(-distance, dim=-2).mean(dim=-1)
            all_out_class.append(out_class)
            if get_uncertainty:
                all_uncertainty_class.append(self.get_class_uncertainty(pred_concept_sigma, self.head[0].weight))
        out_dict['all_pred_class_prob'] = all_out_class
        if get_uncertainty:
            out_dict['all_class_uncertainty'] = all_uncertainty_class

        return out_dict
