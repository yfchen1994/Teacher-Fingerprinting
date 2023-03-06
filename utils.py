# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
import torchvision.models as torch_models
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
#from skimage import io, transform
import PIL
#import pretrainedmodels as ptm
from torch.utils.data import Dataset, DataLoader

DATASET_PATH_RELATIVE = './datasets/'
DATASET_ROOT = os.path.join(os.path.dirname(__file__), DATASET_PATH_RELATIVE)


def check_directory(path):
    """
    Check whether ``path'' exists.
    If not, it will be created.

    Args:
        path: The path to check.
    """
    if not os.path.exists(path):
        # All the missing intermediate directories will be created
        Path(path).mkdir(parents=True)


def load_pretrained_model(model_name, from_torchvision=True, model_path=None):
    """
    Load the pretrained model to fine-tune.

    Args:
        model_name: the Torch model name, i.e., resnet18, alexnet.
        See details at https://pytorch.org/docs/stable/torchvision/models.html
        from_torchvision: whether pretrained model comes from
                          torchvision.models.
        model_path: the local pretrained model file.

    Returns:
        pretrained_model (torch model): Return the pretrained Torch model.
    """
    if from_torchvision:
        pretrained_model = getattr(torch_models, model_name)(pretrained=True)
        pretrained_model = nn.Sequential(
            *list(pretrained_model.children())[:-1])
    else:
        # For the model from other sources, you can implement this part as needed.
        pass

    return pretrained_model


def load_dataset(dataset_name,
                 kwargs={},
                 from_torch=True,
                 transform=None,
                 target_transform=None):
    """
    Load the dataset.

    Args:
        dataset_name: the Torch dataset name, i.e., MNIST.
                      See details at 
                      https://pytorch.org/docs/stable/torchvision/datasets.html
        train: whether load the training or testing datset.
        from_torch: if it is False, the dataset will be loaded from
                    ``model_path''. Defaule to True.
        data_path: the local dataset path.

    Returns:
       The dataset.
    """
    if not transform:
        transform = transforms.ToTensor()

    if dataset_name == 'NoiseDataset':
        dataset_config = {**{'transform': transform}, **kwargs}
        dataset = NoiseDataset(**dataset_config)
        return dataset
    elif dataset_name == 'CelebADataset':
        dataset_config = {**{'transform': transform}, **kwargs}
        dataset = CelebADataset(**dataset_config)
        return dataset
    elif dataset_name == 'Dogs_vs_Cats':
        dataset_config = {**{'transform': transform}, **kwargs}
        dataset = Dogs_vs_Cats(**dataset_config)
        return dataset
    elif dataset_name == 'TinyImageNet':
        dataset_config = {**{'transform': transform}, **kwargs}
        dataset = TinyImageNet(**dataset_config)
        return dataset

    if from_torch:
        basic_config = {
            'root': DATASET_ROOT,
            'download': True,
            'transform': transform
        }
        dataset_config = {**basic_config, **kwargs}
        if target_transform:
            dataset_config = {
                **dataset_config,
                **{
                    'target_transform': target_transform
                }
            }
        dataset = getattr(torch_datasets, dataset_name)(**dataset_config)
    else:
        # For user-defined dataset, the first thing is to present them in the stardard pytorch
        # dataset format. You can implement this part as needed.
        pass

    return dataset


class NoiseDataset(Dataset):
    """
    Dataset composed of noises.
    """

    def __init__(self, noise_size, dataset_size=2000, transform=None):
        self.noise_size = noise_size
        self.transform = transform
        self.dataset_size = dataset_size
        self._generate_noises()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        samples = self.noises[idx]
        if self.transform:
            samples = self.transform(samples)
        return samples, samples

    def _generate_noises(self):
        noises = np.random.randn(*([
            self.dataset_size,
        ] + list(self.noise_size)))
        noises -= noises.min((1, 2, 3), keepdims=True)
        noises /= noises.max((1, 2, 3), keepdims=True)
        noises = noises * 255.
        noises = noises.astype(np.uint8)
        self.noises = []
        for noise in noises:
            self.noises.append(PIL.Image.fromarray(noise, 'RGB'))


class CelebADataset(Dataset):
    """
    Self-defined CeleBa dataset
    (Note: we don't use the CelebA dataset provided by the
     torchvision package.)
    """
    CSV_FILE = './datasets/celeba/list_attr_celeba.csv'
    ROOT_DIR = './datasets/celeba/img_align_celeba/'

    def __init__(self,
                 csv_file=CSV_FILE,
                 root_dir=ROOT_DIR,
                 split='train',
                 transform=None):
        self.TRAIN_SIZE = 50000
        self.TEST_SIZE = 10000

        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        if self.split == 'train':
            self.dataset_size = self.TRAIN_SIZE
        elif self.split == 'test':
            self.dataset_size = self.TEST_SIZE

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.split == 'train':
            idx = idx % self.TRAIN_SIZE
        elif self.split == 'test':
            idx = self.TRAIN_SIZE + idx % self.TEST_SIZE

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = PIL.Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        attrs = self.data_info.iloc[idx, 1:]
        attrs = np.array(attrs)
        attrs = ((attrs.reshape(-1) + 1) * 0.5).astype('float32')
        return image, torch.from_numpy(attrs).float()


class Dogs_vs_Cats(Dataset):
    """
    dogs-vs-cats dataset
    (https://www.kaggle.com/c/dogs-vs-cats/data)
    """
    CSV_FILE_NAME = 'data_info.csv'
    ROOT_DIR = './datasets/dogs_vs_cats/'

    def __init__(self,
                 csv_file_name=CSV_FILE_NAME,
                 root_dir=ROOT_DIR,
                 split='train',
                 transform=None):

        self.TRAIN_SIZE = 20000
        self.TEST_SIZE = 5000

        self.transform = transform
        self.split = split

        self.root_dir = os.path.join(root_dir, self.split)
        csv_file = os.path.join(self.root_dir, csv_file_name)
        self.data_info = pd.read_csv(csv_file)
        self.dataset_size = self.data_info.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])
        image = PIL.Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = int(self.data_info.iloc[idx, 1])
        return image, label


class TinyImageNet(Dataset):
    ROOT_DIR = './datasets/tiny-imagenet/images/'

    def __init__(self, root_dir=ROOT_DIR, transform=None):
        self.dataset_size = 20000
        self.root_dir = root_dir
        self.transform = transform
        self.all_imgs = os.listdir(self.root_dir)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.all_imgs[idx])

        image = PIL.Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


class CustomImageFolder(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.all_imgs = os.listdir(self.img_dir)
        self.all_imgs.sort()

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.all_imgs[idx])
        image = PIL.Image.open(img_loc)

        if self.transform:
            image = self.transform(image)
        return image


def save_model(model, model_path):
    """
    Save the model.

    Args:
        model: the model to save.
        model_path: where to save the model.
    """
    #folder = ('/').join(model_path.split('/')[:-1])
    folder = os.path.dirname(model_path)
    check_directory(folder)
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path):
    """Load the model weights.

    Args:
        model: the PyTorch model. 
        model_path: where the weights are saved.

    Returns:
        The model with trained weights.
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model
    else:
        raise ValueError(
            'Model weight path {} does not exist!'.format(model_path))


def load_config(config_file):
    """
    Read the configuration file (YAML format).

    Args:
        config_file: The configuration file path.

    Returns:
        A dictionary contains the configuration.
    """
    if not os.path.exists(config_file):
        raise ValueError(
            "Configuration file {} does not exist!".format(config_file))
    # Read the configuration file.
    with open(config_file) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs
