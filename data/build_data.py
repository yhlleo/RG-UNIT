
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.distributed as distributed

from .celeba_dataset import CelebA, CelebA_test
from .cat2dog_dataset import Cat2Dog, Cat2Dog_test

def build_loader(
    dataset_train,
    dataset_test,
    batch_size=16,
    num_workers=4,
    pin_memory=True
):
    num_tasks = distributed.get_world_size()
    global_rank = distributed.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, 
        num_replicas=num_tasks, 
        rank=global_rank, 
        shuffle=True
    )
    train_loader = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return train_loader, test_loader

def build_dataset(
    image_dir, 
    train_list_path,
    test_list_path,
    crop_size=178,
    img_size=128,
    dataset_name='CelebA',
    is_retrieval=False
):
    transform_train = build_transform(crop_size, img_size, True, dataset_name)
    transform_test  = build_transform(crop_size, img_size, False, dataset_name)
    if dataset_name == "CelebA":
        dataset_train = CelebA(
            image_dir, 
            train_list_path,
            test_list_path,
            transform_train,
            True,
            is_retrieval
        )

        dataset_test = CelebA(
            image_dir, 
            train_list_path,
            test_list_path,
            transform_test,
            False,
            is_retrieval
        )
    else:
        dataset_train = Cat2Dog(
            image_dir, 
            train_list_path,
            test_list_path,
            transform_train,
            True
        )

        dataset_test = Cat2Dog(
            image_dir, 
            train_list_path,
            test_list_path,
            transform_test,
            False
        )
    return dataset_train, dataset_test


def build_transform(
    crop_size=178, 
    img_size=128,
    is_train=True,
    dataset_name='CelebA'
):
    transform = []     
    if dataset_name == 'CelebA':
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(img_size))
    else:
        transform.append(T.Resize((img_size, img_size)))

    if is_train:
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.ColorJitter(brightness=.005, contrast=.005))

    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform


def build_ret_transform(
    crop_size=178, 
    img_size=128,
    dataset_name='CelebA'
):
    transform = []     
    if dataset_name == 'CelebA':
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(img_size))
    else:
        transform.append(T.Resize((img_size, img_size)))

    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform


def build_ret_loader(
    image_dir, 
    train_list_path,
    crop_size=178,
    img_size=128,
    dataset_name='CelebA',
    batch_size=16,
    pin_memory=True,
    num_workers=4
): 
    transform = build_ret_transform(crop_size, img_size, dataset_name)
    if dataset_name == "CelebA":
        dataset = CelebA_test(
            image_dir, 
            train_list_path, 
            None, 
            transform=transform, 
            is_train=True
        )
    else:
        dataset = Cat2Dog_test(
            image_dir, 
            train_list_path, 
            None, 
            transform=transform, 
            is_train=True
        )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return data_loader
