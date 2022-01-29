# Code Ref: https://github.com/tomlawrenceuk/GTSRB-Dataloader/blob/master/gtsrb_dataset.py
# Download dataset from https://onedrive.live.com/?authkey=%21AKNpIXu0xpmVm1I&cid=25B382439BAD237F&id=25B382439BAD237F%21224763&parId=25B382439BAD237F%21224762&action=locate
# Unzip the code and make the path the root_dir below.

import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):
        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

        self.imgs = []
        self.labels = []

        for idx in range(len(self.csv_data)):
            img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                    self.csv_data.iloc[idx, 0])
            img = Image.open(img_path)
            classId = self.csv_data.iloc[idx, 1]
            self.labels.append(classId)

            if self.transform is not None:
                img = self.transform(img)

            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


class PoisonedGTSRB(Dataset):
    def __init__(self, root,
                 train=True,
                 poison_ratio=0.1,
                 target=0,
                 patch_size=5,
                 random_loc=False,
                 upper_right=True,
                 bottom_left=False,
                 augmentation=True, 
                 black_trigger=False):
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root

        if random_loc:
            print('Using random location')
        if upper_right:
            print('Using fixed location of Upper Right')
        if bottom_left:
            print('Using fixed location of Bottom Left')

        # init trigger
        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(), lambda x: x * 255]
        )
        trigger = Image.open("./dataset/triggers/htbd.png").convert("RGB")
        if black_trigger:
            print('Using black trigger')
            trigger = Image.open("./dataset/triggers/clbd.png").convert("RGB")
        trigger = trans_trigger(trigger) # [3,5,5] [0,255]
        # trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0)))  # 5,5,3 [0,255]

        if self.train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.3403, 0.3121, 0.3214),
                                        (0.2724, 0.2608, 0.2669))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.3403, 0.3121, 0.3214),
                                        (0.2724, 0.2608, 0.2669))
            ])


        trans_imgs = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), lambda x: x * 255]
        )
        if self.train:
            dataset = GTSRB(root, train=True, transform=trans_imgs)
        else:
            dataset = GTSRB(root, train=False, transform=trans_imgs)

        self.imgs = dataset.imgs
        self.labels = dataset.labels
        image_size = self.imgs.shape[-1]

        np.random.seed(100)
        index = np.random.permutation(len(self.imgs))
        sub_index = index[:int(len(self.imgs) * poison_ratio)]

        for i in sub_index:
            if random_loc:
                start_x = random.randint(0, image_size - patch_size)
                start_y = random.randint(0, image_size - patch_size)
            elif upper_right:
                start_x = image_size - patch_size - 3
                start_y = image_size - patch_size - 3
            elif bottom_left:
                start_x = 3
                start_y = 3
            else:
                assert False

            self.imgs[i][:, start_x: start_x + patch_size, start_y: start_y + patch_size] = trigger
            self.labels[i] = target
            
    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)




if __name__ == "__main__":
    # Create Datasets
    trainset = PoisonedGTSRB(root='./data', train=True, transform=None)
    testset = PoisonedGTSRB(root='./data', train=False, transform=None)

    # Load Datasets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)
