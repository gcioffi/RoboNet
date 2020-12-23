# Visualize some images from the dataset.
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch 

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_images(dataset_dir):
    images = []

    # Open the HDF5 file
    f = h5py.File(dataset_dir, "r+")
    images = np.array(f["/images"]).astype("float32")
    return images


def plot_images(images):
    nrows = 5
    ncols = 5
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(24,4))
    
    idxs = np.random.randint(0, images.shape[0], nrows*ncols)

    # plot test images
    for cnt, idx in enumerate(idxs):
        ax = fig.add_subplot(nrows, ncols, cnt+1, xticks=[], yticks=[])
        image = np.transpose(images[idx], (1, 2, 0))
        if image.shape[2] == 1:
            image = image.squeeze()
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)

    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Visualize some images from dataset.")
    parser.add_argument('--dataset', type=str, help="Path to the dataset (hdf5 file).")
    args = parser.parse_args()

    images = load_images(args.dataset)
    print('# images loaded:')
    print(images.shape)

    plot_images(images)

