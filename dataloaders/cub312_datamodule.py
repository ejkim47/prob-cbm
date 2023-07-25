import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from pdb import set_trace as stop
import os, random
from dataloaders.cub312_dataset import CUBDataset
import warnings


warnings.filterwarnings("ignore")


def get_data(args):
    data_root = args.dataroot
    pkl_root = args.metadataroot
    batch_size = args.batch_size
    resol = args.img_size

    # This is zero-indexed (https://github.com/yewsiang/ConceptBottleneck/issues/15)
    attr2attrlabel = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91,
                      93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181,
                      183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
                      254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

    with open('./datasets/class_attr_data_10/attributes.txt', 'r') as f:
        strings = f.readlines()

    attr_group_dict = {}
    attr_group_dict_name = {}
    for i, idx in enumerate(attr2attrlabel):
        label = strings[idx].split(' ')[-1].replace('\n', '')
        group = label.split('::')[0]
        if group in attr_group_dict.keys():
            attr_group_dict[group].append(i)
            attr_group_dict_name[group].append(label)
        else:
            attr_group_dict[group] = [i]
            attr_group_dict_name[group] = [label]
    
    # print(attr_group_dict_name)
    # print(attr_group_dict)

    workers = args.workers
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

    group2concept = torch.zeros(len(attr_group_dict), len(attr2attrlabel))
    for g_idx, (g_name, g) in enumerate(attr_group_dict.items()):
        for c in g:
            group2concept[g_idx, c] = 1

    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    train_dataset, val_dataset, test_dataset = None, None, None
    drop_last = True
    resized_resol = int(resol * 256 / 224)

    trainTransform = transforms.Compose([
        transforms.ColorJitter(brightness=32 / 255, saturation=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((resized_resol, resized_resol)),
        transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])

    testTransform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])

    cub_root = os.path.join(data_root, 'CUB_200_2011')
    image_dir = os.path.join(cub_root, 'images')
    train_list = os.path.join(pkl_root, 'train.pkl')
    val_list = os.path.join(pkl_root, 'val.pkl')
    test_list = os.path.join(pkl_root, 'test.pkl')

    train_dataset = CUBDataset(image_dir, train_list, trainTransform, attr_group_dict=attr_group_dict, testing=False)

    image_dir = os.path.join(cub_root, 'images')
    val_dataset = CUBDataset(image_dir, val_list, testTransform, 
                                attr_group_dict=attr_group_dict, testing=True)
    test_dataset = CUBDataset(image_dir, test_list, testTransform, 
                                attr_group_dict=attr_group_dict, testing=True)


    train_loader, val_loader, test_loader = None, None, None
    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                  drop_last=drop_last)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, train_dataset.class2concept, group2concept
