# Test the Convolutional Autoencoder for a specific robot.
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from robonet.cae.dataset import get_dataset


def visualize(test_images, reconstructed_images):
    nrows = 2
    ncols = 5
    n_imgs_to_vis = np.minimum(10, len(test_images))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, \
        figsize=(24,4))
    # plot n test images
    for idx in np.arange(n_imgs_to_vis):
        ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])
        test_image = np.transpose(test_images[idx], (1, 2, 0))
        if test_image.shape[2] == 1:
            test_image = test_image.squeeze()
        plt.imshow(test_image, cmap='gray', vmin=0, vmax=1)
        #ax.set_title('GT')
        fig.suptitle('Ground truth')

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, \
        figsize=(24,4))
    # plot n reconstructed images
    for idx in np.arange(n_imgs_to_vis):
        ax = fig.add_subplot(nrows, ncols, idx+1, xticks=[], yticks=[])
        reconstructed_image = np.transpose(reconstructed_images[idx], (1, 2, 0))
        if reconstructed_image.shape[2] == 1:
            reconstructed_image = reconstructed_image.squeeze()
        plt.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
        #ax.set_title('Reconstructed')
        fig.suptitle('Output')

    plt.show()


def run(data_dir, model_dir, batch_size, cae_version, plot):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s" % device)

    test_loader = get_dataset(data_dir, 'test', batch_size)

    # Check if rgb or grayscale
    grayscale = False
    dataiter = iter(test_loader)
    batch = dataiter.next()
    batch = batch['image'].numpy()
    if batch.shape[1] == 1:
        grayscale = True

    if cae_version == 1:
        from robonet.cae.model import ConvAutoencoder_v1
        CAE = ConvAutoencoder_v1(grayscale).to(device)
    elif cae_version == 2:
        from robonet.cae.model import ConvAutoencoder_v2
        CAE = ConvAutoencoder_v2(grayscale).to(device)
    CAE.load_state_dict(torch.load(model_dir))
    print("Loaded CAE:")
    for param_tensor in CAE.state_dict():
        print(param_tensor, "\t", CAE.state_dict()[param_tensor].size())
    
    CAE.eval()

    #loss_fun = nn.BCELoss()
    loss_fun = nn.L1Loss()

    n_imgs_to_vis = 10
    test_images_to_vis = []
    reconstructed_images_to_vis = []

    test_loss = 0.0
    for i, batch in enumerate(test_loader):
        images = batch['image'].to(device) 
        outputs = CAE(images)
        loss = loss_fun(outputs, images)
        test_loss += loss.item()*images.size(0)

        if plot and i < n_imgs_to_vis:
            idx = np.random.randint(0, batch_size)
            #idx = 0
            
            test_image_to_vis = images.cpu().numpy()[idx]
            test_images_to_vis.append(test_image_to_vis)
            
            h = np.shape(test_image_to_vis)[1]
            w = np.shape(test_image_to_vis)[2]
            c = np.shape(test_image_to_vis)[0]
            reconstructed_images = outputs.view(batch_size, c, h, w)
            reconstructed_images = reconstructed_images.detach().cpu().numpy()
            reconstructed_image_to_vis = reconstructed_images[idx]
            reconstructed_images_to_vis.append(reconstructed_image_to_vis)

    test_loss = test_loss/len(test_loader)
    print('\nTest Loss: {:.6f}\n'.format(test_loss))

    if plot:
        visualize(test_images_to_vis, reconstructed_images_to_vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a CAE for a \
    specific robot sequence.")
    parser.add_argument('--data_dir', type=str, help="Folder containing test data.")
    parser.add_argument('--model_dir', type=str, help="Path to model.")
    parser.add_argument('--batch_size', type=int, help="Batch size.")
    parser.add_argument('--cae_version', type=int, help="CAE version.")
    parser.add_argument('--plot', type=bool, help="Plot.")
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.model_dir)
    run(args.data_dir, args.model_dir, args.batch_size, args.cae_version, args.plot)

