
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
import pickle
from pdb import set_trace as stop
from PIL import Image
import pandas as pd

from config import DEBUG


class CUBDataset(Dataset):
    def __init__(self, img_dir, img_list, image_transform, attr_group_dict=None, testing=False):
        with open(img_list, "rb" ) as f:
            self.labels = pickle.load(f)

        self.image_transform = image_transform
        self.img_dir = img_dir
        self.num_concepts = 112
        self.num_labels = 200

        # np.random.seed()
        self.attr_group_dict = attr_group_dict

        self.testing = testing
        self.epoch = 1
        self.class2concept= self._get_class2concept()
        self.concept_imb_ratio = self._get_concept_imbalance_ratio()

        self.bbox = self._load_bbox()


    def _get_class2concept(self):
        class2concept = torch.zeros(200, 112)
        for label in self.labels:
            class2concept[label['class_label']] = torch.Tensor(label['attribute_label'])
        class2concept[class2concept == 0] = -1
        return class2concept
    
    def _get_concept_imbalance_ratio(self):
        num_attr = torch.zeros(112)
        for label in self.labels:
            num_attr += torch.Tensor(label['attribute_label'])
        imbalance_ratio = len(self.labels) / num_attr - 1
        return imbalance_ratio

    def _load_bbox(self):
        root = self.img_dir[:self.img_dir.index('CUB_200_2011')]
        bbox = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ', names=['img_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        bbox.img_id = bbox.img_id.astype(int)
        return bbox

    def __getitem__(self, index):
        name = self.labels[index]['img_path']
        if 'images' in name:
            name = name.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/images/' ,'')
        img_path = os.path.join(self.img_dir, name)

        image = Image.open(img_path).convert('RGB')

        if self.image_transform is not None:
            image = self.image_transform(image)

        concept = torch.Tensor(self.labels[index]['attribute_label'])
        class_label = torch.Tensor([self.labels[index]['class_label']])
        concept_certainty = torch.Tensor(self.labels[index]['attribute_certainty'])

        sample = {}
        sample['image'] = image
        sample['concept_label'] = concept
        sample['class_label'] = class_label
        sample['concept_certainty'] = concept_certainty
        sample['imageID'] = name

        return sample

    def find_image(self, fname):
        for index, label in enumerate(self.labels):
            name = label['img_path']
            if 'images' in name:
                name = name.split('/')[-1]
            if name == fname:
                return index

    def get_image(self, fname):
        index = self.find_image(fname)
        return self.__getitem__(index)

    def __len__(self):
        if DEBUG:
            return int(len(self.labels) * 0.1)
        return len(self.labels)
