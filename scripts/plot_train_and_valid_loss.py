# Visualize train and validation loss.
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch 

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot(train_loss, valid_loss):
    plt.plot(np.arange(1, train_loss.shape[0]+1, 1), train_loss, 'b', \
        label="train loss")
    plt.plot(np.arange(100, train_loss.shape[0]+1, 100), valid_loss, 'r', \
        label="valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Visualize train and validation loss.")
    parser.add_argument('--train_loss_fn', type=str, help="Path to the .txt containg training loss.")
    parser.add_argument('--valid_loss_fn', type=str, help="Path to the .txt containg valid. loss.")
    args = parser.parse_args()

    assert(args.train_loss_fn)
    assert(args.valid_loss_fn)
    train_loss = np.loadtxt(args.train_loss_fn)
    valid_loss = np.loadtxt(args.valid_loss_fn)

    print("Loaded %d training loss datapoints." % train_loss.shape[0])
    print("Loaded %d validation loss datapoints." % valid_loss.shape[0])

    plot(train_loss, valid_loss)

