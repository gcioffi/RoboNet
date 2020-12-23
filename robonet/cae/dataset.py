# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch
# Code inspired by: 
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import io
import os

import cv2
import h5py
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RobotManipulatorDataset(Dataset):
    def __init__(self, data_dir, dataset_type, transform=None):
        self.dataset_dir = os.path.join(data_dir, dataset_type + '.hdf5')
        self.transform = transform

    def __len__(self):
        f = h5py.File(self.dataset_dir, "r+")
        images = np.array(f["/images"]).astype("float32")
        return images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        f = h5py.File(self.dataset_dir, "r+")
        # images: N X C X H X W
        # N: num samples, C: channels, H: height, W: width
        images = np.array(f["/images"]).astype("float32")

        sample = {'image': images[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        return {'image': torch.from_numpy(image.copy())}


def get_dataset(data_dir, dataset_type, batch_size, shuffle=True):
    transform = ToTensor()
    dataset = RobotManipulatorDataset(data_dir, dataset_type, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=0)
    return data_loader

