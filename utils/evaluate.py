import torch
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.metrics import average_precision_score, recall_score, precision_score, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from tqdm import tqdm
from config import DEBUG

warnings.filterwarnings("ignore")
sns.set()

def get_concept_label(dataset, data=None):
    if dataset == 'cub':

        attr2attrlabel = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91,
                        93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181,
                        183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
                        254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
        
        with open('./datasets/class_attr_data_10/attributes.txt', 'r') as f:
            strings = f.readlines()

        concept_label_list = []
        for i, idx in enumerate(attr2attrlabel):
            concept_label_list.append(strings[idx].split(' ')[-1].replace('\n', ''))
    
    elif dataset == 'awa2':
        with open('./datasets/awa2/predicates.txt', 'r') as f:
            strings = f.readlines()
        concept_label_list = []
        if data is not None:
            strings = [strings[c_idx] for c_idx in data.concept_list]
        for string in strings:
            label = string.split('\t')[-1].replace('\n', '')
            concept_label_list.append(label)
    elif 'mnist' in dataset:
        concept_label_list = [str(i) for i in range(10)]
    
    return concept_label_list

def get_class_label(dataset):
    if dataset == 'cub':
        with open('./datasets/class_attr_data_10/attributes.txt', 'r') as f:
            strings = f.readlines()

        # concept_label_list = []
        # for i, idx in enumerate(attr2attrlabel):
        #     concept_label_list.append(strings[idx].split(' ')[-1].replace('\n', ''))
        class_label_list = np.arange(200).tolist()
    elif dataset == 'awa2':
        with open('/home/eunji/Datasets/Animals_with_Attributes2/classes.txt', 'r') as f:
            strings = f.readlines()
        class_label_list = []
        for string in strings:
            label = string.split('\t')[-1].replace('\n', '')
            class_label_list.append(label)
    elif 'mnist' in dataset:
        class_label_list = np.arange(12).tolist()

    return class_label_list


def get_inv_normalize(dataset):
    if dataset in ['cub', 'awa2']:
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif 'mnist' in dataset:
        mean, std = [0.5], [1.0]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )

    return inv_normalize

def get_normalize(dataset):
    if dataset in ['cub', 'awa2']:
        mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif 'mnist' in dataset:
        mean, std = [0.5], [1.0]
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )

    return normalize

def compute_metrics(args, all_predictions, all_targets, loss, elapsed,
                    all_metrics=False, verbose=True, epoch=0, split='test'):
    concept_acc = 0
    class_acc = 0

    if args.pred_concept:
        all_preds_concepts = all_predictions[:, :args.num_concepts].clone()
        all_targets_concepts = all_targets[:, :args.num_concepts].clone()

        all_preds_concepts_binary = all_preds_concepts.clone()
        all_preds_concepts_binary[all_preds_concepts >= 0.5] = 1
        all_preds_concepts_binary[all_preds_concepts < 0.5] = 0

        concept_accs = []
        for i in range(all_preds_concepts_binary.size(1)):
            concept_accs.append(accuracy_score(all_targets_concepts[:, i], all_preds_concepts_binary[:, i]))
        concept_acc = np.array(concept_accs).mean()

        avg_prec = precision_score(all_targets_concepts, all_preds_concepts_binary, average="macro")
        avg_recall = recall_score(all_targets_concepts, all_preds_concepts_binary, average="macro")
        concept_f1 = f1_score(all_targets_concepts, all_preds_concepts_binary, average="macro")
        all_confusion = []
        for c in range(all_targets_concepts.shape[1]):
            tn, fp, fn, tp = confusion_matrix(all_targets_concepts[:, c], all_preds_concepts_binary[:, c], labels=[0, 1]).ravel()
            all_confusion.append([tn, fp, fn, tp])
        all_confusion = np.array(all_confusion)

        avg_tnr = (all_confusion[:, 0] / (all_confusion[:, 0] + all_confusion[:, 1])).mean()

    if args.pred_class:
        s_idx = args.num_concepts if args.pred_concept else 0
        all_preds_classes = all_predictions[:, s_idx:].clone()
        all_targets_classes = all_targets[:, s_idx:].clone()
        pred_max_val, pred_max_idx = torch.max(all_preds_classes, 1)
        _, target_max_idx = torch.max(all_targets_classes, 1)

        class_acc = (pred_max_idx == target_max_idx).sum().item() / pred_max_idx.size(0)

        _top_max_k_vals, top_max_k_inds = torch.topk(
            all_preds_classes, 3, dim=1, largest=True, sorted=True
        )

        class_acc_top3 = (top_max_k_inds == target_max_idx.unsqueeze(-1)).sum().item() / target_max_idx.size(0)


    if verbose and loss is not None:
        print('loss:  {:0.3f}'.format(loss))
        print('----')

    metrics_dict = {}

    print('Concept Acc:    {:0.3f} \t Concept F1:    {:0.3f}'.format(concept_acc, concept_f1))
    print('Class Acc:    {:0.3f}'.format(class_acc))
    if args.pred_concept:
        metrics_dict['concept_acc'] = concept_acc
        metrics_dict['concept_f1'] = concept_f1
        metrics_dict['concept_prec'] = avg_prec
        metrics_dict['concept_recall'] = avg_recall
        metrics_dict['concept_tnr'] = avg_tnr
    if args.pred_class:
        metrics_dict['class_acc'] = class_acc

    print('')

    return metrics_dict


mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)


def run_cutout_test(args, model, data, desc, device):
    print("Run cutout test for diff patch size")
    model.eval()

    patch_size_list = [64]

    all_uncertainties = torch.zeros(len(data.dataset) * 2, args.num_concepts, len(patch_size_list) + 1).cpu()
    all_uncertainties_class = torch.zeros(len(data.dataset) * 2, len(patch_size_list) + 1).cpu()

    end_idx = 0
    for batch in tqdm(data, mininterval=0.5, desc=desc, leave=False, ncols=50):

        images = batch['image'].float().to(device)
        B = images.shape[0]
        start_idx, end_idx = end_idx, end_idx + B

        with torch.no_grad():
            preds_dict, losses_dict = model(images, T=args.n_samples_inference)
            all_uncertainties[start_idx:end_idx, ..., 0] = preds_dict['pred_concept_uncertainty'].data.cpu()
            if 'pred_class_uncertainty' in preds_dict.keys():
                all_uncertainties_class[start_idx:end_idx, 0] = preds_dict['pred_class_uncertainty'].data.cpu()

            for p_idx, patch_size in enumerate(patch_size_list):
                cut_images = batch['image'].float().clone()
                x = y = cut_images.shape[-1] // 2 - patch_size // 2
                cut_images[:, :, x:x + patch_size, y:y + patch_size] = 0
                cut_preds_dict, _ = model(cut_images.to(device), T=args.n_samples_inference)
                all_uncertainties[start_idx:end_idx, ..., p_idx + 1] = cut_preds_dict[
                    'pred_concept_uncertainty'].data.cpu()
                if 'pred_class_uncertainty' in cut_preds_dict.keys():
                    all_uncertainties_class[start_idx:end_idx, p_idx + 1] = cut_preds_dict['pred_class_uncertainty'].data.cpu()

    all_uncertainties = all_uncertainties[:end_idx]
    all_uncertainties_class = all_uncertainties_class[:end_idx]

    all_uncertainties_avg = all_uncertainties.mean(dim=-2)  # avg over concepts

    concept_increase = all_uncertainties_avg.permute(1, 0)[1:] - all_uncertainties_avg.permute(1, 0)[:1]
    print('Concept: ' + ' '.join([str(v.item()) for v in (concept_increase > 0).float().mean(dim=-1)]))
    class_increase = all_uncertainties_class.permute(1, 0)[1:] - all_uncertainties_class.permute(1, 0)[:1]
    print('Class: ' + ' '.join([str(v.item()) for v in (class_increase > 0).float().mean(dim=-1)]))
    