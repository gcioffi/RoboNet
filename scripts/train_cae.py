# Train a Convolutional Autoencoder for a specific robot.
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch

import argparse
from datetime import datetime
import io
import os
from pathlib import Path as path_lib
import timeit
import warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from robonet.cae.dataset import get_dataset
from robonet.datasets import load_metadata


LEARNING_RATE = 0.001


def run(train_dir, valid_dir, cae_version, batch_size, n_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s" % device)

    train_loader = get_dataset(train_dir, 'train', batch_size)
    validation_loader = get_dataset(valid_dir, 'valid', batch_size)
    
    # Check if rgb or grayscale
    grayscale = False
    dataiter = iter(train_loader)
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
    print('--- CAE model ---')
    print(CAE)

    # loss function
    #loss_fun = nn.BCELoss()
    loss_fun = nn.L1Loss()
    optimizer = torch.optim.Adam(CAE.parameters(), lr=LEARNING_RATE)

    # Save dir
    out_dir = os.path.join(os.path.dirname(args.train_dir), 'cae_model')
    path_lib(out_dir).mkdir(parents=True, exist_ok=True)

    # training
    t_start = timeit.default_timer()
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            # forward pass
            CAE.train()
            outputs = CAE(images)
            # calculate the loss
            loss = loss_fun(outputs, images)
            # backward pass
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            train_loss += loss.item()*images.size(0)
                
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        # eval. on validation dataset
        if epoch%100 == 0:
            with torch.no_grad():
                valid_loss = 0.0
                for val_batch in validation_loader:
                    images = val_batch['image'].to(device)                  
                    CAE.eval()
                    outputs = CAE(images)
                    loss = loss_fun(outputs, images)
                    valid_loss += loss.item()*images.size(0)

                valid_loss = valid_loss/len(validation_loader)
                print('\nEpoch: {} \tValidation Loss: {:.6f}\n'.format(epoch, valid_loss))
        
        # Save intermediate results
        if epoch%1000 == 0:
            out_fn = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pt'
            if grayscale:
                out_fn = 'cae_version_' + str(cae_version) +  '_grayscale_' + out_fn
            else:
                out_fn = 'cae_version_' + str(cae_version) + '_' + out_fn
            out_fn = 'epochs_' + str(epoch) + '_' + out_fn
            out_fn = os.path.join(out_dir, out_fn)
            torch.save(CAE.state_dict(), out_fn)
            print('Model saved: %s' % out_fn)

    print('\nTraining time: {:.2f} [min]'.format((timeit.default_timer()-t_start)*1/60.0))

    # save
    out_fn = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pt'
    if grayscale:
        out_fn = 'cae_version_' + str(cae_version) +  '_grayscale_' + out_fn
    else:
        out_fn = 'cae_version_' + str(cae_version) + '_' + out_fn
    out_fn = 'epochs_' + str(n_epochs) + '_' + out_fn
    out_fn = os.path.join(out_dir, out_fn)
    torch.save(CAE.state_dict(), out_fn)
    print('Model saved: %s' % out_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CAE for a \
    specific robot sequence.")
    parser.add_argument('--cae_version', type=int, help="Cae version.")
    parser.add_argument('--train_dir', type=str, \
        help="Folder containing training data.")
    parser.add_argument('--valid_dir', type=str, \
        help="Folder containing validation data.")
    parser.add_argument('--batch_size', type=int, help="Batch size.")
    parser.add_argument('--n_epochs', type=int, help="Num. epochs.")
    args = parser.parse_args()

    assert os.path.exists(args.train_dir)
    assert os.path.exists(args.valid_dir)
    run(args.train_dir, args.valid_dir, args.cae_version, args.batch_size, \
        args.n_epochs)

