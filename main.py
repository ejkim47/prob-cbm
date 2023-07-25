import torch
import torch.nn as nn
import os, shutil

from adamp import AdamP

from models import *
from config import DEBUG, get_configs
import utils.evaluate as evaluate
from utils.util import get_class, load_saved_model
from run_epoch import run_epoch

args = get_configs()

print("\n=========================================")
print("Arguments")
for arg in vars(args):
    print(f'{arg}: \t{getattr(args, arg)}')
print("=========================================")
if 'wandb' in args.log_tool:
    import wandb

# Change how to set device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())
    print('')
else:
    device = torch.device('cpu')
    print('Current device: cpu')

class2concept, group2concept = None, None
if args.dataset == 'cub':
    from dataloaders.cub312_datamodule import get_data
    loaders, class2concept, group2concept = get_data(args)
args.group2concept = group2concept

print('Labels: {}'.format(args.num_classes))
print('Concepts: {}'.format(args.num_concepts))

backbone = get_class(args.backbone, 'models')
backbone_type = 'resnet'

model = get_class(args.model_type, 'models')(
    use_probabilsitic_concept=args.use_probabilsitic_concept,
    backbone=backbone, num_concepts=args.num_concepts, num_classes=args.num_classes, 
    pretrained=args.pretrained, pred_class=args.pred_class, hidden_dim=args.hidden_dim, token2concept=group2concept, class_hidden_dim=args.class_hidden_dim,
    class2concept=class2concept, 
    use_scale=args.use_scale, use_neg_concept=args.use_neg_concept, train_class_mode=args.train_class_mode,
    activation_concept2class=args.activation_concept2class, n_samples_inference=args.n_samples_inference,
    init_shift=5, init_negative_scale=5, intervention_prob=args.intervention_prob)

# print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("# of params: ", pytorch_total_params)
model = model.to(device)

loss_weight = {'concept': args.loss_weight_concept, 'class': args.loss_weight_class}

if args.only_eval:
    log_dir = args.config.replace(args.config.split('/')[-1], '')
    model_path = os.path.join(log_dir, 'last_checkpoint.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'last_checkpoint_class.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(log_dir, 'last_checkpoint_concept.pth')
    model, epoch = load_saved_model(model_path, model, device="cuda")
    print("Load model ", model_path)
    if loaders['test'] is not None:
        data_loader = loaders['test']
    elif loaders['val'] is not None:
        data_loader = loaders['val']
    else:
        data_loader = loaders['train']

    exit(0)

if args.train_class_mode in ['joint', 'independent']:
    stages = ['joint']
elif args.train_class_mode == 'sequential':
    stages = ['concept', 'class']

epoch = 0
for stage in stages:
    # Set optimizer
    params_new, params_new_nodecay, params_old, params_old_nodecay = [], [], [], []
    params_new_name, params_new_nodecay_name, params_old_name, params_old_nodecay_name = [], [], [], []

    no_weight_decay = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else []
    params_to_train = [name for name, _ in model.named_parameters()]
    if stage == 'class':
        params_to_train = [name for name, _ in model.named_parameters() if name in model.params_to_classify()]
        # print('Train only class params: ', params_to_train)
    if stage == 'concept':
        params_to_train = [name for name, _ in model.named_parameters() if name not in model.params_to_classify()]
        # print('Train only class params: ', params_to_train)
    for name, parameter in model.named_parameters():
        if name in params_to_train and parameter.requires_grad and name.split('.')[0] != 'features':
            if name.split('.')[0] in ['head', 'concept_vectors', 'concept_vectors_logsigma', 'class_mean', 'stem', 'class_negative_scale',
                                        'class_head', 'concept_head', 'shift', 'negative_scale', 'scale', 'mean_head', 'logsigma_head']:
                if name.split('.')[0] in no_weight_decay:
                    params_new_nodecay.append(parameter)
                    params_new_nodecay_name.append(name)
                else:
                    params_new.append(parameter)
                    params_new_name.append(name)
            else:
                if name.split('.')[0] in no_weight_decay:
                    params_old_nodecay.append(parameter)
                    params_old_nodecay_name.append(name)
                else:
                    params_old.append(parameter)
                    params_old_name.append(name)
        # else:
        #     print(f"Miss {name} in training!")


    if args.optim == 'adam':
        optimizer = torch.optim.Adam([{'params': params_old, 'lr': args.lr, 'weight_decay': getattr(args, 'weight_decay', 0)},
                                        {'params': params_old_nodecay, 'lr': args.lr, 'weight_decay': 0},
                                        {'params': params_new, 'lr': args.lr * args.lr_ratio, 'weight_decay': getattr(args, 'weight_decay', 0)},
                                        {'params': params_new_nodecay, 'lr': args.lr * args.lr_ratio, 'weight_decay': 0}])
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD([{'params': params_old, 'lr': args.lr, 'weight_decay': getattr(args, 'weight_decay', 1e-4)},
                                        {'params': params_old_nodecay, 'lr': args.lr, 'weight_decay': 0},
                                        {'params': params_new, 'lr': args.lr * args.lr_ratio, 'weight_decay': getattr(args, 'weight_decay', 1e-4)},
                                        {'params': params_new_nodecay, 'lr': args.lr * args.lr_ratio, 'weight_decay': 0}], momentum=0.9)
    elif args.optim == 'adamp':
        optimizer = AdamP([{'params': params_old, 'lr': args.lr, 'weight_decay': getattr(args, 'weight_decay', 0)},
                                        {'params': params_old_nodecay, 'lr': args.lr, 'weight_decay': 0},
                                        {'params': params_new, 'lr': args.lr * args.lr_ratio, 'weight_decay': getattr(args, 'weight_decay', 0)},
                                        {'params': params_new_nodecay, 'lr': args.lr * args.lr_ratio, 'weight_decay': 0}], betas=(0.9, 0.999), eps=1e-8)

    if args.scheduler_type == 'cosineannealing':
        step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_class if stage == 'class' else args.epochs)
    else:
        step_scheduler = None

    start_epoch = 0
    if stage != 'class':
        if args.resume:
            log_dir = args.config.replace(args.config.split('/')[-1], '')
            model_path = os.path.join(log_dir, f'last_checkpoint_{stage}.pth')
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('Optimizer not loaded!')
            start_epoch = checkpoint['epoch']
            epoch = start_epoch
            print("Load model", model_path)

    test_start = getattr(args, 'test_start', args.epochs - 10)

    freeze_old_params = False
    val_best_acc = 0
    for stage_epoch in range(start_epoch + 1, args.epochs_class + 1 if stage == 'class' else args.epochs + 1):
        epoch += 1
        print('======================== {} ========================'.format(epoch))
        for param_group in optimizer.param_groups:
            print('LR: {}'.format(param_group['lr']))

        loaders['train'].dataset.epoch = epoch
        ################### Train #################
        warm = True if epoch < args.warm_epochs + 1 else False

        all_preds, all_targs, all_certs, all_cls_certs, train_loss = run_epoch(args, model, loaders['train'],
                                                    optimizer, epoch, 'Training', device=device, loss_weight=loss_weight,
                                                    train=True, warm=warm, stage=stage)
        train_metrics = evaluate.compute_metrics(args, all_preds, all_targs, train_loss['total'], 0)
        if 'wandb' in args.log_tool:
            wandb.log({f'lr_group{idx}': param_group['lr'] for idx, param_group in enumerate(optimizer.param_groups)}, step=epoch)
            wandb.log({f'loss_train/{k}_loss': v for k, v in train_loss.items()}, step=epoch)
            wandb.log({f'acc/train_{k}': v for k, v in train_metrics.items()}, step=epoch)

        ################### Valid #################
        all_preds, all_targs, all_certs, all_cls_certs, valid_loss = run_epoch(args, model, loaders['val'], None, epoch, 'Validating', device=device, loss_weight=loss_weight, warm=warm)
        valid_metrics = evaluate.compute_metrics(args, all_preds, all_targs, valid_loss['total'], 0)
        # evaluate.run_cutout_test(args, model, loaders['val'], 'Cutout Test', device=device)
        if 'wandb' in args.log_tool:
            wandb.log({f'loss_val/{k}_loss': v for k, v in valid_loss.items()}, step=epoch)
            wandb.log({f'acc/val_{k}': v for k, v in valid_metrics.items()}, step=epoch)

        ################### Test #################
        if loaders['test'] is not None and epoch >= test_start:
            all_preds, all_targs, all_certs, all_cls_certs, test_loss = run_epoch(args, model, loaders['test'], None, epoch, 'Testing', device=device, loss_weight=loss_weight, warm=warm)
            test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, test_loss['total'], 0)
            evaluate.run_cutout_test(args, model, loaders['test'], 'Cutout Test', device=device)
            if 'wandb' in args.log_tool:
                wandb.log({f'acc/test_{k}': v for k, v in test_metrics.items()}, step=epoch)
        else:
            test_loss, test_metrics = valid_loss, valid_metrics

        if step_scheduler is not None:
            if args.scheduler_type in ['cosineannealing']:
                step_scheduler.step(stage_epoch)
            elif args.scheduler_type == 'plateau':
                step_scheduler.step(valid_loss['total'])

        if not DEBUG:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                    os.path.join(args.log_folder, f'last_checkpoint_{stage}.pth'))
            if stage == 'class' and val_best_acc < valid_metrics['class_acc']:
                val_best_acc = valid_metrics['class_acc']
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(args.log_folder, f'best_checkpoint_{stage}.pth'))


############## Log and Save ##############
all_preds, all_targs, all_certs, all_cls_certs, test_loss = run_epoch(args, model, loaders['test'], None, 0, 'Testing', device=device, loss_weight=loss_weight)
test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_certs, all_cls_certs, test_loss['total'], 0)