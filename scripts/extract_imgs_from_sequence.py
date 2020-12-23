# Extract imageas from a sequence (images contained in a single .hdf5)
# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch 

import argparse
import io
import os
from pathlib import Path as path_ut
import warnings
warnings.filterwarnings("ignore")

import cv2
import h5py
import hashlib
import imageio
import numpy as np

from robonet.datasets import load_metadata


# Default loader parameters
IMG_SIZE = [112, 112]


# Similar as in hdf5_loader.py
def load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims, start_time=0, n_load=None):
    cam_group = file_pointer['env']['cam{}_video'.format(cam_index)]
    old_dims = file_metadata['frame_dim']
    length = file_metadata['img_T']
    encoding = file_metadata['img_encoding']
    image_format = file_metadata['image_format']

    #print("Image original dim (%d, %d)" % (old_dims[0], old_dims[1]))
    #print("Image target dim (%d, %d)" % (target_dims[0], target_dims[1]))
    #print("Num. of images: %d" % length)

    if n_load is None:
        n_load = length

    old_height, old_width = old_dims
    target_height, target_width = target_dims
    resize_method = cv2.INTER_CUBIC
    if target_height * target_width < old_height * old_width:
        resize_method = cv2.INTER_AREA
    
    images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
    if encoding == 'mp4':
        buf = io.BytesIO(cam_group['frames'][:].tostring())
        img_buffer = [img for t, img in enumerate(imageio.get_reader(buf, format='mp4')) if start_time <= t < n_load + start_time]
    elif encoding == 'jpg':
        img_buffer = [cv2.imdecode(cam_group['frame{}'.format(t)][:], cv2.IMREAD_COLOR)[:, :, ::-1] 
                                for t in range(start_time, start_time + n_load)]
    else: 
        raise ValueError("encoding not supported")
    
    for t, img in enumerate(img_buffer):
        if (old_height, old_width) == (target_height, target_width):
            images[t] = img
        else:
            images[t] = cv2.resize(img, (target_width, target_height), interpolation=resize_method)
    
    if image_format == 'RGB':
        return images
    elif image_format == 'BGR':
        return images[:, :, :, ::-1]
    raise NotImplementedError


def load_images(fn, img_size=IMG_SIZE):
    # sequence = h5py.File(path_to_hdf5, 'r')
    assert os.path.exists(fn) and os.path.isfile(fn), "invalid fn"
    assert 'hdf5' in fn
    data_folder = '/'.join(fn.split('/')[:-1])
    meta_data = load_metadata(data_folder)
    file_metadata = meta_data.get_file_metadata(fn)

    with open(fn, 'rb') as f:
        buf = f.read()
    assert hashlib.sha256(buf).hexdigest() == file_metadata['sha256'], "file hash doesn't match meta-data. maybe delete pkl and re-generate?"

    with h5py.File(io.BytesIO(buf)) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"        
        cam_index = 0
        images = []
        images.append(load_camera_imgs(cam_index, hf, file_metadata, img_size, start_time, n_states)[None])
        images = np.swapaxes(np.concatenate(images, 0), 0, 1)

    return images


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Visualize a sequence of images contained in a .hdf5")
    parser.add_argument('--fn', type=str, help="Path to the hdf5 file.")
    parser.add_argument('--out_dir', type=str, help="Folder where to save images.", 
    default='../dataset/data')
    args = parser.parse_args()

    images = load_images(args.fn)
    print('Loaded images', images.shape)

    # Debug visualization
    image = images[0, 0, :, :, :]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow('image rgb', image_bgr)
    cv2.imshow('image gray', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_dir = os.path.join(args.out_dir, os.path.basename(args.fn)[:-5])
    out_dir = os.path.join(out_dir, 'images')
    path_ut(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        img = cv2.cvtColor(images[i, 0, :, :, :], cv2.COLOR_RGB2GRAY)
        img_name = os.path.join(out_dir, '%04d.png' % i)
        cv2.imwrite(img_name, img)
    print('Saved %d images to %s' % (images.shape[0], out_dir))

