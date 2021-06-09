# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data

from .utils import load_cat2dog_files as load_files

import random
random.seed(1234)

class Cat2Dog(data.Dataset):
    """Dataset class for the Cat2Dog dataset."""
    def __init__(self, 
        image_dir,         # image data path
        train_list_path,   # training image list, including both image name and labels
        test_list_path,    # testing image list, including both image name and labels
        transform=None, 
        is_train=True
    ):
        """Initialize and preprocess the Cat2Dog dataset."""
        self.image_dir    = image_dir
        self.transform    = transform
        self.is_retrieval = is_retrieval

        if is_train:
            self.dataset = load_files(train_list_path, "train" if is_train else "test")
        else:
            self.dataset = load_files(test_list_path, "train" if is_train else "test")
        self.num_images  = len(self.dataset)

        self.cats, self.dogs = self.split_animals()

    def split_animals(self):
        cats, dogs = [], []
        for da in self.dataset:
            if da[1]:
                dogs.append(da)
            else:
                cats.append(da)
        return cats, dogs

    def __getitem__(self, index):
        # The outputs rely on the `is_retrieval`
        fname_src, src_label = self.dataset[index]
        if src_label: # if dog
            fname_trg, trg_label = random.choice(self.cats)
        else:
            fname_trg, trg_label = random.choice(self.dogs)

        src_image = Image.open(os.path.join(self.image_dir, fname_src)).convert("RGB")
        trg_image = Image.open(os.path.join(self.image_dir, fname_trg)).convert("RGB")

        src_image, trg_image = self.transform(src_image), self.transform(trg_image)
        src_label = torch.tensor(src_label).long()
        trg_label = torch.tensor(trg_label).long()
        return  src_image, trg_image, src_label, trg_label 

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Cat2Dog_test(data.Dataset):
    def __init__(self, 
        image_dir, 
        train_list_path, 
        test_list_path, 
        transform=None, 
        is_train=False
    ):
        self.image_dir = image_dir
        self.transform = transform

        if is_train:
            self.dataset = load_files(train_list_path, "train" if is_train else "test")
        else:
            self.dataset  = load_files(test_list_path, "train" if is_train else "test")

        self.num_images = len(self.test_dataset)

    def __getitem__(self, index):
        fname_src, src_label = self.dataset[index]
        trg_label = 1 - src_label

        image = Image.open(os.path.join(self.image_dir, fname_src)).convert("RGB")
        image = self.transform(image)

        trg_label = torch.tensor(trg_label).long()
        src_label = torch.tensor(src_label).long()
        return image, fname_src, trg_label, src_label

    def __len__(self):
        """Return the number of images."""
        return self.num_images