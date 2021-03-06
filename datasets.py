import os
import glob
import torch
import random
import numpy as np
from shutil import move
import kornia.augmentation as A
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, STL10, CelebA, ImageFolder
import torchvision.datasets as datasets

__all__ = ['mnist_dataloader', 'cifar10_dataloader', 'cifar100_dataloader', 'tiny_imagenet_dataloader',
           'svhn_dataloader', 'stl10_dataloader', 'celeba_dataloader', 'imagenet_dataloader', 'PostTensorTransform']


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)


def imagenet_dataloader(batch_size=128, data_dir='/data'):
    data_dir = os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC')
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    resize_transform = []
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(resize_transform + [
            transforms.RandomResizedCrop(288),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(resize_transform + [
            transforms.CenterCrop(288),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(mean=torch.tensor([0.485, 0.456, 0.406]),
                                                      std=torch.tensor([0.229, 0.224, 0.225]))

    return train_loader, val_loader, val_loader, dataset_normalization


def celeba_dataloader(batch_size=64, data_dir='./data'):
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = CelebA(data_dir, "train", transform=train_transform, download=True)
    val_set = CelebA(data_dir, "valid", transform=test_transform, download=True)
    test_set = CelebA(data_dir, "test", transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, None


def mnist_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.1):
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_size = int(60000 * (1 - val_ratio))
    val_size = 60000 - train_size

    train_set = Subset(MNIST(data_dir, train=True, transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(MNIST(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = MNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def cifar10_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(50000 * (1 - val_ratio))
    val_size = 50000 - train_size

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    return train_loader, val_loader, test_loader, dataset_normalization


def cifar100_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(50000 * (1 - val_ratio))
    val_size = 50000 - train_size

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True),
                       list(range(train_size)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    return train_loader, val_loader, test_loader, dataset_normalization


def tiny_imagenet_dataloader(batch_size=64, data_dir='./data/tiny_imagenet/', permutation_seed=10):
    """
    Prepare for the Tiny-ImageNet dataset
    Step 1: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    Step 2: unzip -qq 'tiny-imagenet-200.zip'
    Step 3: rm tiny-imagenet-200.zip (optional)
    Code primarily from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/val_format.py
    Args:
        batch_size:
        data_dir:
        permutation_seed:
    Returns:
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_path = os.path.join(data_dir, 'train/')
    val_path = os.path.join(data_dir, 'val/')
    test_path = os.path.join(data_dir, 'test/')

    if os.path.exists(os.path.join(val_path, "images")):
        if os.path.exists(test_path):
            os.rename(test_path, os.path.join(data_dir, "test_original"))
            os.mkdir(test_path)
        val_dict = {}
        val_anno_path = os.path.join(val_path, "val_annotations.txt")
        with open(val_anno_path, 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]

        paths = glob.glob('./tiny-imagenet-200/val/images/*')
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(val_path + str(folder)):
                os.mkdir(val_path + str(folder))
                os.mkdir(val_path + str(folder) + '/images')
            if not os.path.exists(test_path + str(folder)):
                os.mkdir(test_path + str(folder))
                os.mkdir(test_path + str(folder) + '/images')

        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if len(glob.glob(val_path + str(folder) + '/images/*')) < 25:
                dest = val_path + str(folder) + '/images/' + str(file)
            else:
                dest = test_path + str(folder) + '/images/' + str(file)
            move(path, dest)

        os.rmdir(os.path.join(val_path, "images"))

    np.random.seed(permutation_seed)

    train_set = Subset(ImageFolder(train_path, transform=train_transform), range(100000))
    val_set = Subset(ImageFolder(val_path, transform=test_transform), range(10000))
    test_set = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return train_loader, val_loader, test_loader, dataset_normalization


def svhn_dataloader(batch_size=64, data_dir='./data/', val_ratio=0.0):
    num_workers = 2
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_size = int(73257 * (1 - val_ratio))
    val_size = 73257 - train_size

    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True), list(range(train_size)))
    val_set = Subset(SVHN(data_dir, split='train', transform=test_transform, download=True),
                     list(range(train_size, train_size + val_size)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                             drop_last=True)
    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])

    return train_loader, val_loader, test_loader, dataset_normalization


def stl10_dataloader(batch_size=64, data_dir='./data/'):
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = STL10(data_dir, split='train', download=True, transform=train_transform)
    test_set = STL10(data_dir, split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4467, 0.4398, 0.4066], std=[0.2242, 0.2215, 0.2239])
    return train_loader, test_loader, test_loader, dataset_normalization


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, image_size, random_crop, random_rotation, dataset):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((image_size, image_size), padding=random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(random_rotation), p=0.5)
        if dataset == "CIFAR10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x