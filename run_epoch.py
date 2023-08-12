import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from utils.util import get_class

from models import *
from utils.loss import MCBCELoss



def run_epoch(args, model, data, optimizer, epoch, desc, device, loss_weight=None, train=False,  warm=False, inference_with_sampling=False, stage='joint'):
    if train:
        model.train()
        if isinstance(model, (ProbCBM)):
            if warm and hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = False
            elif hasattr(model, 'cnn_module'):
                for p in model.cnn_module.parameters():
                    p.requires_grad = True
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()
    all_certainties = torch.zeros(len(data.dataset) * 2, args.num_concepts).cpu()
    all_cls_certainties = torch.zeros(len(data.dataset) * 2).cpu()
    all_targets = torch.zeros(len(data.dataset) * 2, args.num_labels).cpu()

    batch_idx = 0
    end_idx = 0
    loss_tot_dict = {'total': 0}

    # Set criterion for class and concept
    criterion_class = getattr(args, 'criterion_class', 'ce')
    if criterion_class == 'ce':
        criterion_class = nn.CrossEntropyLoss()
    else:
        raise ValueError('Got criterion_class', criterion_class)
    
    criterion_concept = getattr(args, 'criterion_concept', 'bce')
    if criterion_concept == 'bce':
        criterion_concept = nn.BCEWithLogitsLoss()
    elif criterion_concept == 'bce_prob':
        criterion_concept = nn.BCELoss()
    elif criterion_concept == 'MCBCELoss':
        in_criterion = nn.BCELoss(reduction='none')
        criterion_concept = get_class(criterion_concept, 'utils.loss')(criterion=in_criterion, reduction='mean', vib_beta=args.vib_beta, \
            group2concept=args.group2concept)
    else:
        raise ValueError('Got criterion_concept', criterion_concept)

    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['image'].float().to(device)
        target_class = batch['class_label'][:, 0].long().to(device)
        target_concept = batch['concept_label'].float().to(device)

        if train:
            preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, T=args.n_samples_train, stage=stage)
        else:
            with torch.no_grad():
                preds_dict, losses_dict = model(images, target_concept=target_concept, target_class=target_class, inference_with_sampling=inference_with_sampling, T=args.n_samples_inference)

        B = images.shape[0]
        class_label_onehot, concept_labels, labels, concept_uncertainty, class_uncertainty = None, None, None, None, None
        if args.pred_class:
            class_labels = batch['class_label'].float()
            class_label_onehot = torch.zeros(class_labels.size(0), args.num_classes)
            class_label_onehot.scatter_(1, class_labels.long(), 1)
            labels = class_label_onehot

        concept_labels = batch['concept_label'].float()
        if args.pred_concept:
            if labels is not None:
                labels = torch.cat((concept_labels, labels), 1)
            else:
                labels = concept_labels
        assert (labels is not None)

        loss, pred = 0, None
        loss_iter_dict = {}
        if args.pred_concept:
            if isinstance(criterion_concept, MCBCELoss):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept, concept_loss_dict = criterion_concept(\
                    probs=preds_dict['pred_concept_prob'],
                    image_mean=preds_dict['pred_mean'], image_logsigma=preds_dict['pred_logsigma'],
                    concept_labels=target_concept, negative_scale=preds_dict['negative_scale'], shift=preds_dict['shift'])
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
                for k, v in concept_loss_dict.items():
                    if k != 'loss':
                        loss_iter_dict['pcme_' + k] = v
            elif isinstance(criterion_concept, (nn.BCELoss)):
                pred_concept = preds_dict['pred_concept_prob']
                loss_concept = criterion_concept(pred_concept, target_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']
            else:
                pred_concept = preds_dict['pred_concept_logit']
                loss_concept = criterion_concept(pred_concept, target_concept)
                pred_concept = torch.sigmoid(pred_concept)
                if 'pred_concept_uncertainty' in preds_dict.keys():
                    concept_uncertainty = preds_dict['pred_concept_uncertainty']

            if stage != 'class':
                loss += loss_concept * loss_weight['concept']
            pred = pred_concept
            loss_iter_dict['concept'] = loss_concept

        if args.pred_class:
            if 'pred_class_logit' in preds_dict.keys():
                pred_class = preds_dict['pred_class_logit']
                loss_class = criterion_class(pred_class, target_class)
                pred_class = F.softmax(pred_class, dim=-1)
            else:
                assert 'pred_class_prob' in preds_dict.keys()
                pred_class = preds_dict['pred_class_prob']
                loss_class = F.nll_loss(pred_class.log(), target_class, reduction='mean')
            loss_iter_dict['class'] = loss_class

            if stage != 'concept':
                loss += loss_class * loss_weight['class']
            pred = pred_class if pred is None else torch.cat((pred_concept, pred_class), dim=1)

            if 'pred_class_uncertainty' in preds_dict.keys():
                class_uncertainty = preds_dict['pred_class_uncertainty']

        for k, v in losses_dict.items():
            loss_iter_dict[k] = v
            if k in loss_weight.keys() and loss_weight[k] != 0:
                loss += v * loss_weight[k]
        loss_out = loss

        for k, v in loss_iter_dict.items():
            if v != v:
                print(k, v)

        if train:
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()

        ## Updates ##
        loss_tot_dict['total'] += loss_out.item()
        for k, v in loss_iter_dict.items():
            if k not in loss_tot_dict.keys():
                try:
                    loss_tot_dict[k] = v.item()
                except:
                    loss_tot_dict[k] = v
            else:
                try:
                    loss_tot_dict[k] += v.item()
                except:
                    loss_tot_dict[k] += v
        start_idx, end_idx = end_idx, end_idx + B

        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0), -1)

        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()
        if concept_uncertainty is not None:
            all_certainties[start_idx:end_idx] = concept_uncertainty.data.cpu()
        if class_uncertainty is not None:
            all_cls_certainties[start_idx:end_idx] = class_uncertainty.data.cpu()
        batch_idx += 1

    for k, v in loss_tot_dict.items():
        loss_tot_dict[k] = v / batch_idx


    return all_predictions[:end_idx], all_targets[:end_idx], all_certainties[:end_idx], all_cls_certainties[:end_idx], loss_tot_dict

