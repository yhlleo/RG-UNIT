from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

def get_loader(image_dir, crop_size=178, image_size=128, batch_size=16, 
    train_list_path=None, test_list_path=None, dataset_name='CelebA', 
    mode='train', num_workers=4):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    if 'CelebA' in dataset_name:
        transform.append(T.CenterCrop(crop_size))
        transform.append(T.Resize(image_size))
    else:
        transform.append(T.Resize((image_size, image_size)))
        transform.append(T.CenterCrop(crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset_name == "CelebA":
        from data_ios.celeba_data import CelebA
        cur_dataset = CelebA(image_dir, train_list_path, test_list_path, transform, mode)
    elif dataset_name == 'CelebA_retrieval':
        from data_ios.celeba_data_retrieval import CelebA_retrieval
        cur_dataset = CelebA_retrieval(image_dir, train_list_path, test_list_path, transform, mode)
    elif dataset_name == 'CelebA_retrieval_test':
        from data_ios.celeba_data_retrieval_test import CelebA_retrieval_test
        cur_dataset = CelebA_retrieval_test(image_dir, train_list_path, test_list_path, transform, mode)
    elif dataset_name == "Cat2Dog":
        from data_ios.cat2dog_data import Cat2Dog
        cur_dataset = Cat2Dog(image_dir, train_list_path, test_list_path, transform, mode)
    elif dataset_name == "Cat2Dog_retrieval":
        from data_ios.cat2dog_data_retrieval import Cat2Dog_retrieval
        cur_dataset = Cat2Dog_retrieval(image_dir, train_list_path, test_list_path, transform, mode)
    elif dataset_name == "Cat2Dog_retrieval_test":
        from data_ios.cat2dog_data_retrieval_test import Cat2Dog_retrieval_test
        cur_dataset = Cat2Dog_retrieval_test(image_dir, train_list_path, test_list_path, transform, mode)
    else:
        assert False, "Unknow dataset name."
    data_loader = data.DataLoader(dataset=cur_dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
