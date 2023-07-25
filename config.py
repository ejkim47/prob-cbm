
import argparse
import os
from os.path import join as ospj
import shutil
import datetime
import dateutil.tz
import yaml
import wandb
import copy, re

from utils.util import set_random_seed


DEBUG = False


def change2abspath(path):
    if path[0] != '/':
        current_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(current_path, path)
    return path


def configure_data_type(args):
    if args.dataset == 'cub':
        args.num_classes = 200
        args.num_concepts = getattr(args, 'num_concepts', 112)
    elif args.dataset == 'awa2':
        args.num_classes = 50
        args.num_concepts = getattr(args, 'num_concepts', 85 if not args.reduce_concept else 45)
    elif args.dataset == 'toymnist':
        args.num_classes = 12
        args.num_concepts = getattr(args, 'num_concepts', 10)
    else:
        raise ValueError("dataset should be one of ['cub', 'awa2', 'toymnist']")

    num_labels = 0
    if args.pred_class:
        num_labels += args.num_classes
    if args.pred_concept:
        num_labels += args.num_concepts
    args.num_labels = num_labels

def configure_data_folder(args):
    args.dataroot = change2abspath(args.dataroot)
    args.metadataroot = change2abspath(args.metadataroot)


def configure_log_folder(args):
    log_folder = change2abspath(args.log_dir)
    if args.only_eval:
        log_folder = args.config.replace(args.config.split('/')[-1], '')
        assert ('config' not in log_folder)
    elif args.resume:
        args.experiment_name = args.config.split('/')[-2]
        log_folder = args.config.replace(args.config.split('/')[-1], '')
        old_log_folder = log_folder
    else:
        if 'time' in args.experiment_name:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d-%H-%M')
            args.experiment_name = args.experiment_name.replace('time', timestamp)
        if 'modeltype' in args.experiment_name:
            modelv = re.sub(r'[^0-9]', '', args.model_type)
            modeln = args.model_type.replace(modelv, '')
            args.experiment_name = args.experiment_name.replace('modeltype', f'{modeln}{modelv}')
        log_folder = ospj(log_folder, 'train_log', args.dataset, args.experiment_name)

    if args.resume:
        assert (os.path.isdir(log_folder))
        print("Start from last_checkpoint in", log_folder)
    elif args.only_eval:
        print(log_folder)
        assert (os.path.isdir(log_folder))
        print("Inference with last_checkpoint and best_checkpoint in", log_folder)
    elif os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
            os.makedirs(log_folder)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    else:
        os.makedirs(log_folder)
    return log_folder


def configure_wandb(args, wandbid=None):
    new_args = vars(copy.deepcopy(args))
    for k in ['only_eval', 'override_cache', 'log_tool']:
        del new_args[k]

    if 'wandb' in args.log_tool:
        prj_name = "ProbCBM"
        if args.resume and (os.path.exists(ospj(args.log_folder, 'wandbid.txt')) or wandbid is not None):
            if wandbid is None:
                with open(ospj(args.log_folder, 'wandbid.txt'), 'r') as f:
                    wandbid = f.readline()
            wandb.init(project=prj_name, config=new_args, resume="must", id=wandbid)
        else:
            wandb.init(project=prj_name, config=new_args)
            with open(ospj(args.log_folder, 'wandbid.txt'), 'w') as f:
                f.write('{}'.format(wandb.run.id))
        wandb.run.name = f'{args.experiment_name}'
        wandb.run.save()


def configure_config(args):
    # args.config and args.log_folder should be absolute path
    if not args.only_eval and not args.resume:
        with open(args.config, 'rb') as f:
            conf_new = yaml.load(f.read(), Loader=yaml.Loader)
        for k, v in conf_new.items():
            if getattr(args, k) is not None and getattr(args, k) != v:
                conf_new[k] = getattr(args, k)
        yaml.dump(conf_new, open(os.path.join(args.log_folder, args.config.split('/')[-1]), 'w'), sort_keys=False)

        wandb.save(os.path.join(args.log_folder, args.config.split('/')[-1]))
        

def configure_arguments(args):
    if not args.pred_class:
        args.loss_weight_class = 0.
    if not args.pred_concept:
        args.loss_weight_concept = 0.


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--config', type=str, default='configs/config_exp.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--only_eval', action='store_true', default=False)

    flags = parser.parse_args()

    del_attr = []
    for flag in vars(flags):
        if getattr(flags, flag) is None:
            del_attr.append(flag)
    for flag in del_attr:
        delattr(flags, flag)

    if DEBUG:
        print("DEBUG MODE")

    with open(change2abspath('configs/config_base.yaml'), 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)  # load the config file

    # Restore from wandb
    if flags.config.split('.')[-1] != 'yaml':
        api = wandb.Api()
        run = api.run(flags.config)
        exp_name = run.name
        conf['experiment_name'] = exp_name

        exp_file_name = None
        for file in run.files():
            if 'config_exp' in file.name or file.name in ['last_checkpoint_no-opt.pth', 'last_checkpoint.pth']:
                file.download(root=ospj(conf['log_dir'], 'restore', exp_name), replace=True)
                exp_file_name = file.name if 'config_exp' in file.name else exp_file_name
        flags.config = ospj(conf['log_dir'], 'restore', exp_name, exp_file_name)
        wandb.finish()

    flags.config = change2abspath(flags.config)
    print(f"Load config file: {flags.config}")
    with open(flags.config, 'rb') as f:
        conf_new = yaml.load(f.read(), Loader=yaml.Loader)
    conf.update(conf_new)
    conf.update(vars(flags))
    args = argparse.Namespace(**conf)

    if DEBUG:
        args.workers = 0
        args.override_cache = True
        args.experiment_name = 'test'
        args.log_tool = []
    if args.only_eval:
        args.log_tool = []

    configure_data_folder(args)
    configure_data_type(args)
    if not DEBUG:
        configure_arguments(args)
        args.log_folder = configure_log_folder(args)
        configure_wandb(args)
        configure_config(args)

    set_random_seed(args.seed)

    return args
