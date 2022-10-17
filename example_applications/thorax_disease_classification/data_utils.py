import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms as T,models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid

pathology_list = [
        'Cardiomegaly',
        'Emphysema',
        'Effusion',
        'Hernia',
        'Nodule',
        'Pneumothorax',
        'Atelectasis',
        'Pleural_Thickening',
        'Mass',
        'Edema',
        'Consolidation',
        'Infiltration',
        'Fibrosis',
        'Pneumonia'
        ]

class NIH_Dataset(Dataset):

    def __init__(
        self, 
        data, 
        img_dir, 
        transform=None
        ):

        self.data = data
        self.img_dir = img_dir 
        self.transform = transform 

    def __len__(
        self
        ):

        return len(self.data)

    def __getitem__(
        self, 
        idx
        ):
        img_file = self.img_dir + self.data.iloc[:,0][idx]
        img = Image.open(img_file).convert('RGB')
        label = np.array(self.data.iloc[:,1:].iloc[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

def inv_data_transform(img):
    img = img.permute(1,2,0)
    img = img * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
    return img

def get_fl_nih_subset(
    num_clients: int,
    batch_size: int,
    num_workers: int = 1,
    drop_none: bool = True
    ):

    _, _, _, trainset, data_df, pathology_list = get_nih_subset(
        batch_size = batch_size,
        num_workers = num_workers,
        drop_none = drop_none
    )

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader, data_df, pathology_list

def get_nih_subset(
    batch_size,
    train_frac=0.8,
    num_workers=1,
    drop_none=True
    ):

    data_df = pd.read_csv('./dataset/NIH/sample/sample_labels.csv')

    for pathology in pathology_list :
        data_df[pathology] = data_df['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

    data_df['No Findings'] = data_df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

    data_df = data_df.drop(list(data_df.iloc[:,1:11].columns.values), axis = 1)

    # Drop all patients with no findings
    if drop_none:
        data_df = data_df.drop(['No Findings'], axis = 1) 

    data_transform = T.Compose([
        T.RandomRotation((-20,+20)),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                    ])

    trainds = NIH_Dataset(
        data_df,
        img_dir = './dataset/NIH/sample/images/',
        transform = data_transform
        )

    len_ds = len(data_df)
    len_train_ids = math.ceil(len_ds*train_frac)
    len_valid_ids = math.ceil((len_ds - math.ceil(( len_ds * train_frac ))) / 2)
    len_test_ids = math.floor((len_ds - math.ceil(( len_ds * train_frac ))) / 2)

    trainset, validset, testset = random_split(
        trainds,
        [
            len_train_ids,
            len_valid_ids,
            len_test_ids
            ]
        )
    
    trainloader = DataLoader(
        trainset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
        )

    validloader = DataLoader(
        validset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
        )

    testloader = DataLoader(
        testset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
        )

    return trainloader, validloader, testloader, trainset, data_df, pathology_list

if __name__ == '__main__':
    # trainloader, validloader, testloader = get_nih_subset(
    #     batch_size=32,
    #     train_frac=0.8
    # )

    num_clients = 100
    batch_size = 32
    num_workers = 1
    fraction_fit = 0.1
    fraction_evaluate = 0.2
    min_fit_clients = 10
    min_evaluate_clients = 20
    num_rounds = 10

    trainloaders, valloaders, testloader = get_fl_nih_subset(
        num_clients = num_clients,
        batch_size = batch_size
        )

    print(len(trainloaders))