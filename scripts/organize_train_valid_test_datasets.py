# Organize train, validation, and test datasets for a specific robot.
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch 

import argparse
import os
from pathlib import Path as path_lib

import cv2
import h5py
import numpy as np

from extract_imgs_from_sequence import load_images

PERC_TRAIN = 0.7
PERC_VALID = 0.1
PERC_TEST = 0.2

IMG_SIZE = [112, 112]


def organize_dataset(dir, robot):
    # sequence = h5py.File(path_to_hdf5, 'r')
    assert os.path.exists(dir), "dataset not found!"
    sequences = os.listdir(dir)
    
    robot_sequences = []
    robot_name_len = len(robot)
    for sequence in sequences:
        if sequence[0] == robot[0] and len(sequence) > robot_name_len and \
            sequence[0:robot_name_len] == robot:
            robot_sequences.append(sequence)
    n = len(robot_sequences)
    print("Found %d sequences of %s" % (n, robot))

    indxs = np.arange(n)
    np.random.shuffle(indxs)

    n_train = int(n * PERC_TRAIN)
    n_valid = int(n * PERC_VALID)
    n_test = int(n * PERC_TEST)

    train_list = []
    valid_list = []
    test_list = []
    for cnt, idx in enumerate(indxs):
        if cnt < n_train:
            train_list.append(robot_sequences[idx])
        elif n_train <= cnt < (n_train + n_valid):
            valid_list.append(robot_sequences[idx])
        elif (n_train + n_valid) <= cnt < (n_train + n_valid + n_test):
            test_list.append(robot_sequences[idx])

    # Read images in array of dimension (C,H,W) (same convention as pytorch)
    train_images = []
    train_images_grayscale = []

    for seq in train_list:
        fn = os.path.join(dir, seq)
        images = load_images(fn, IMG_SIZE)
        for img in images:
            image_rgb = img[0]
            image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            train_images.append(image_rgb.transpose((2, 0, 1)))
            train_images_grayscale.append(image_grayscale[np.newaxis, :, :])
    
    train_images = np.array(train_images).astype(np.float32)
    train_images_grayscale = np.array(train_images_grayscale).astype(np.float32)
    
    train_images = train_images / 255.
    train_images_grayscale = train_images_grayscale / 255.

    valid_images = []
    valid_images_grayscale = []
    for seq in valid_list:
        fn = os.path.join(dir, seq)
        images = load_images(fn, IMG_SIZE)
        for img in images:
            image_rgb = img[0]
            image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            valid_images.append(image_rgb.transpose((2, 0, 1)))
            valid_images_grayscale.append(image_grayscale[np.newaxis, :, :])
    valid_images = np.array(valid_images).astype(np.float32)
    valid_images_grayscale = np.array(valid_images_grayscale).astype(np.float32)
    valid_images = valid_images / 255.
    valid_images_grayscale = valid_images_grayscale / 255.

    test_images = []
    test_images_grayscale = []
    for seq in test_list:
        fn = os.path.join(dir, seq)
        images = load_images(fn, IMG_SIZE)
        for img in images:
            image_rgb = img[0]
            image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            test_images.append(image_rgb.transpose((2, 0, 1)))
            test_images_grayscale.append(image_grayscale[np.newaxis, :, :])
    test_images = np.array(test_images).astype(np.float32)
    test_images_grayscale = np.array(test_images_grayscale).astype(np.float32)
    test_images = test_images / 255.
    test_images_grayscale = test_images_grayscale / 255.

    return train_list, train_images, train_images_grayscale, \
        valid_list, valid_images, valid_images_grayscale, \
            test_list, test_images, test_images_grayscale


def save_data(out_dir, fn_list, images, dataset, gray_scale):
    if gray_scale:
        out_dir = os.path.join(out_dir, dataset + '_grayscale')
    else:
        out_dir = os.path.join(out_dir, dataset)
    path_lib(out_dir).mkdir(parents=True, exist_ok=True)

    fn_hdf5 = os.path.join(out_dir, dataset + '.hdf5')
    f_hdf5 = h5py.File(fn_hdf5, "w")
    d = f_hdf5.create_dataset(
        "images", np.shape(images), h5py.h5t.IEEE_F32BE, data=images)
    f_hdf5.close()

    fn_txt = os.path.join(out_dir, dataset +  '.txt')
    f_txt = open(fn_txt, "w")
    for fn_seq in fn_list:
        f_txt.write(fn_seq)
        f_txt.write("\n")
    f_txt.close()

    print("Saved %s and %s" % (fn_hdf5, fn_txt))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Split dataset of a specific robot \
    in train, validation, and test.")
    parser.add_argument('--fn', type=str, help="Path to the dataset (hdf5 file).")
    parser.add_argument('--robot', type=str, help="Robot model.", default='stanford_fetch')
    parser.add_argument('--out_dir', type=str, help="Folder where to save train, valid, and test.", 
    default='../dataset/data')
    args = parser.parse_args()

    train_list, train_images, train_images_grayscale, \
        valid_list, valid_images, valid_images_grayscale, \
            test_list, test_images, test_images_grayscale = \
                organize_dataset(args.fn, args.robot)
    print('# train, valid, test sequences: (%d,%d,%d)' % \
    (len(train_list), len(valid_list), len(test_list)))

    out_dir = os.path.join(args.out_dir, args.robot)

    save_data(out_dir, train_list, train_images, 'train_112x112', False)
    save_data(out_dir, train_list, train_images_grayscale, 'train_112x112', True)
    
    save_data(out_dir, valid_list, valid_images, 'valid_112x112', False)
    save_data(out_dir, valid_list, valid_images_grayscale, 'valid_112x112', True)

    save_data(out_dir, test_list, test_images, 'test_112x112', False)
    save_data(out_dir, test_list, test_images_grayscale, 'test_112x112', True)

