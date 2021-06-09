# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data

from .utils import load_celeba_files as load_files

import random
random.seed(1234)

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    def __init__(self, 
        image_dir,         # image data path
        train_list_path,   # training image list, including both image name and labels
        test_list_path,    # testing image list, including both image name and labels
        transform=None, 
        is_train=True, 
        is_retrieval=False # True, used for retrieval; False, used for typical I2I translation
    ):
        self.image_dir    = image_dir
        self.transform    = transform
        self.is_retrieval = is_retrieval

        if is_train:
            self.dataset = load_files(train_list_path, "train" if is_train else "test")
        else:
            self.dataset  = load_files(test_list_path, "train" if is_train else "test")
        self.selected_attrs = 5
        self.num_images = len(self.dataset)

    def __getitem__(self, index):
        # The outputs rely on the `is_retrieval`
        if not self.is_retrieval:
            fname_src, src_label = self.dataset[index]
            fname_trg, trg_label = random.choice(self.dataset)

            src_image = Image.open(os.path.join(self.image_dir, fname_src)).convert("RGB")
            trg_image = Image.open(os.path.join(self.image_dir, fname_trg)).convert("RGB")
            src_image, trg_image = self.transform(src_image), self.transform(trg_image)
            src_label, trg_label = torch.tensor(src_label).float(), torch.tensor(trg_label).float()
            return  src_image, trg_image, src_label, trg_label 
        else:
            fname_src, src_label = self.dataset[index]
            while True:
                fname_neg, neg_label = random.choice(self.dataset)
                if fname_src != fname_neg: break

            # Easy target and negative labels only changing one element
            trg_label = src_label.copy()
            attr_edit_index = random.randint(0,self.selected_attrs-1)
            trg_label[attr_edit_index] = 1-src_label[attr_edit_index] # reverse the attribute in source image

            # sample negative label
            while True:
                _, rnd_label = random.choice(self.dataset)
                if rnd_label != trg_label and rnd_label != src_label: break

            src_image = Image.open(os.path.join(self.image_dir, fname_src)).convert("RGB")
            src_image = self.transform(src_image)

            neg_image = Image.open(os.path.join(self.image_dir, fname_neg)).convert("RGB")
            neg_image = self.transform(neg_image)

            src_label = torch.tensor(src_label).float()
            trg_label = torch.tensor(trg_label).float()
            rnd_label = torch.tensor(rnd_label).float()
            return src_image, neg_image, src_label, trg_label, rnd_label
        

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class CelebA_test(data.Dataset):
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

        # -------------- Random target label from the dataset -------------- #
        trg_idx = random.randint(0,self.num_images-1)
        _, trg_label = self.dataset[trg_idx]
        # -------------------------------------------- #

        src_image = Image.open(os.path.join(self.image_dir, fname_src)).convert("RGB")
        src_image = self.transform(src_image)

        trg_label = torch.tensor(trg_label).float()
        src_label = torch.tensor(src_label).float()
        return src_image, fname_src, trg_label, src_label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

