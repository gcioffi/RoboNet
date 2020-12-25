# Author: Giovanni Cioffi, cioffi@ifi.uzh.ch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder_v1(nn.Module):
    def __init__(self, gray_scale_imgs):
        super(ConvAutoencoder_v1, self).__init__()
        # encoder layers
        if gray_scale_imgs:
            input_chan_conv1 = 1
        else:
            input_chan_conv1 = 3
        output_chan_conv1 = 128
        input_chan_conv2 = output_chan_conv1
        output_chan_conv2 = 96
        input_chan_conv3 = output_chan_conv2
        output_chan_conv3 = 64

        # conv layers
        kernel_size_encoder = 5
        stride_encoder = 1
        padding_encoder = 2

        # conv layers
        self.conv1 = nn.Conv2d(input_chan_conv1, output_chan_conv1, \
            kernel_size=kernel_size_encoder, stride=stride_encoder, \
                padding=padding_encoder)  
        self.conv2 = nn.Conv2d(input_chan_conv2, output_chan_conv2, \
            kernel_size=kernel_size_encoder, stride=stride_encoder, \
                padding=padding_encoder)
        self.conv3 = nn.Conv2d(input_chan_conv3, output_chan_conv3, \
            kernel_size=kernel_size_encoder, stride=stride_encoder, \
                padding=padding_encoder)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # decoder layers
        kernel_size_decoder = 6
        stride_decoder = 2
        padding_decoder = 2

        # transpose conv layers
        self.t_conv1 = nn.ConvTranspose2d(output_chan_conv3, input_chan_conv3, \
            kernel_size=kernel_size_decoder, stride=stride_decoder, \
                padding=padding_decoder)
        self.t_conv2 = nn.ConvTranspose2d(output_chan_conv2, input_chan_conv2, \
            kernel_size=kernel_size_decoder, stride=stride_decoder, \
                padding=padding_decoder)
        self.t_conv3 = nn.ConvTranspose2d(output_chan_conv1, input_chan_conv1, \
            kernel_size=kernel_size_decoder, stride=stride_decoder, \
                padding=padding_decoder)

    def forward(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # decode
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = self.t_conv3(x)
        
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(x)
                
        return x


class ConvAutoencoder_v2(nn.Module):
    def __init__(self, gray_scale_imgs):
        super(ConvAutoencoder_v2, self).__init__()
        # encoder layers
        if gray_scale_imgs:
            input_chan_conv1 = 1
        else:
            input_chan_conv1 = 3
        output_chan_conv1 = 32
        input_chan_conv2 = output_chan_conv1
        output_chan_conv2 = 8

        # conv layers
        kernel_size_encoder = 7
        stride_encoder = 1
        padding_encoder = 3

        # conv layers
        self.conv1 = nn.Conv2d(input_chan_conv1, output_chan_conv1, \
            kernel_size=kernel_size_encoder, stride=stride_encoder, \
                padding=padding_encoder)  
        self.conv2 = nn.Conv2d(input_chan_conv2, output_chan_conv2, \
            kernel_size=kernel_size_encoder, stride=stride_encoder, \
                padding=padding_encoder)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # decoder layers
        # transpose conv
        kernel_size_decoder = 8
        stride_decoder = 2
        padding_decoder = 3

        # transpose conv layers
        self.t_conv1 = nn.ConvTranspose2d(output_chan_conv2, input_chan_conv2, \
            kernel_size=kernel_size_decoder, stride=stride_decoder, \
                padding=padding_decoder)
        self.t_conv2 = nn.ConvTranspose2d(output_chan_conv1, input_chan_conv1, \
            kernel_size=kernel_size_decoder, stride=stride_decoder, \
                padding=padding_decoder)

        # upsampling + convolution
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv3 = nn.Conv2d(output_chan_conv2, input_chan_conv2, \
            kernel_size=kernel_size_encoder, padding=padding_encoder)
        self.conv4 = nn.Conv2d(output_chan_conv1, input_chan_conv1, \
            kernel_size=kernel_size_encoder, padding=padding_encoder)

    def forward(self, x):
        # encode
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # decode
        # use transpose convolution
        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)

        # use upsampling and convolution
        #x = self.upsampling(x)
        #x = F.relu(self.conv3(x))
        #x = self.upsampling(x)
        #x = self.conv4(x)
        
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(x)
                
        return x

