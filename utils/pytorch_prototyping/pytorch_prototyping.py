'''A number of custom pytorch modules with sane defaults that I find useful for model prototyping.'''
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils

import numpy as np

import math
import numbers

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.net(input)


# From https://gist.github.com/wassname/ecd2dac6fc8f9918149853d17e3abf02
class LayerNormConv2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class DownBlock3D(nn.Module):
    '''A 3D convolutional downsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size=4,
                      padding=0,
                      stride=2,
                      bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class UpBlock3D(nn.Module):
    '''A 3D convolutional upsampling block.
    '''

    def __init__(self, in_channels, out_channels, norm=nn.BatchNorm3d):
        super().__init__()

        self.net = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False if norm is not None else True),
        ]

        if norm is not None:
            self.net += [norm(out_channels, affine=True)]

        self.net += [nn.ReLU(True)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class Conv3dSame(torch.nn.Module):
    '''3D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReplicationPad3d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb, ka, kb)),
            nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

    def forward(self, x):
        return self.net(x)


class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
                           bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.ReLU(True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.ReLU(True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels,
                              middle_channels,
                              kernel_size=3,
                              padding=0,
                              stride=1,
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(middle_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        net += [nn.ReflectionPad2d(1),
                nn.Conv2d(middle_channels,
                          out_channels,
                          kernel_size=4,
                          padding=0,
                          stride=2,
                          bias=True if norm is None else False)]

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Unet3d(nn.Module):
    '''A 3d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 norm=nn.BatchNorm3d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet3d."

        # Define the in block
        self.in_layer = [Conv3dSame(in_channels, nf0, kernel_size=3, bias=False)]

        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]

        self.in_layer += [nn.LeakyReLU(0.2, True)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block. The feature map has height and width 1 --> no batchnorm.
        self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    int(min(2 ** (num_down - 1) * nf0, max_channels)),
                                                    norm=None)
        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock3d(int(min(2 ** i * nf0, max_channels)),
                                                        int(min(2 ** (i + 1) * nf0, max_channels)),
                                                        submodule=self.unet_block,
                                                        norm=norm)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv3dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear)]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]
        self.out_layer = nn.Sequential(*self.out_layer)

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class UnetSkipConnectionBlock3d(nn.Module):
    '''Helper class for building a 3D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 norm=nn.BatchNorm3d,
                 submodule=None):
        super().__init__()

        if submodule is None:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     UpBlock3D(inner_nc, outer_nc, norm=norm)]
        else:
            model = [DownBlock3D(outer_nc, inner_nc, norm=norm),
                     submodule,
                     UpBlock3D(2 * inner_nc, outer_nc, norm=norm)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)


class UnetSkipConnectionBlock(nn.Module):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)


    ##ここの編集を忘れずに！
    def forward(self, x):
        # mode = (x.size(-2)//2)%2!=0
        # if mode:
        #     x=F.pad(x,(0,0,1,1),mode="reflect")
        forward_passed = self.model(x)
        res=torch.cat([x, forward_passed], 1)
        # if mode:
        #     res=res[:,:,1:-1,:]
        
        return res


class Unet(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 nf0,
                 num_down,
                 max_channels,
                 use_dropout,
                 upsampling_mode='transpose',
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 outermost_linear=False):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super().__init__()

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv2dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer


class Identity(nn.Module):
    '''Helper module to allow Downsampling and Upsampling nets to default to identity if they receive an empty list.'''

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class DownsamplingNet(nn.Module):
    '''A subnetwork that downsamples a 2D feature map with strided convolutions.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 use_dropout,
                 dropout_prob=0.1,
                 last_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of downsampling steps (each step dowsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param last_layer_one: Whether the output of the last layer will have a spatial size of 1. In that case,
                               the last layer will not have batchnorm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.downs = Identity()
        else:
            self.downs = list()
            self.downs.append(DownBlock(in_channels, per_layer_out_ch[0], use_dropout=use_dropout,
                                        dropout_prob=dropout_prob, middle_channels=per_layer_out_ch[0], norm=norm))
            for i in range(0, len(per_layer_out_ch) - 1):
                if last_layer_one and (i == len(per_layer_out_ch) - 2):
                    norm = None
                self.downs.append(DownBlock(per_layer_out_ch[i],
                                            per_layer_out_ch[i + 1],
                                            dropout_prob=dropout_prob,
                                            use_dropout=use_dropout,
                                            norm=norm))
            self.downs = nn.Sequential(*self.downs)

    def forward(self, input):
        return self.downs(input)

class UpsamplingNet(nn.Module):
    '''A subnetwork that upsamples a 2D feature map with a variety of upsampling options.
    '''

    def __init__(self,
                 per_layer_out_ch,
                 in_channels,
                 upsampling_mode,
                 use_dropout,
                 dropout_prob=0.1,
                 first_layer_one=False,
                 norm=nn.BatchNorm2d):
        '''
        :param per_layer_out_ch: python list of integers. Defines the number of output channels per layer. Length of
                                list defines number of upsampling steps (each step upsamples by factor of 2.)
        :param in_channels: Number of input channels.
        :param upsampling_mode: Mode of upsampling. For documentation, see class "UpBlock"
        :param use_dropout: Whether or not to use dropout.
        :param dropout_prob: Dropout probability.
        :param first_layer_one: Whether the input to the last layer will have a spatial size of 1. In that case,
                               the first layer will not have a norm, else, it will.
        :param norm: Which norm to use. Defaults to BatchNorm.
        '''
        super().__init__()

        if not len(per_layer_out_ch):
            self.ups = Identity()
        else:
            self.ups = list()
            self.ups.append(UpBlock(in_channels,
                                    per_layer_out_ch[0],
                                    use_dropout=use_dropout,
                                    dropout_prob=dropout_prob,
                                    norm=None if first_layer_one else norm,
                                    upsampling_mode=upsampling_mode))
            for i in range(0, len(per_layer_out_ch) - 1):
                self.ups.append(
                    UpBlock(per_layer_out_ch[i],
                            per_layer_out_ch[i + 1],
                            use_dropout=use_dropout,
                            dropout_prob=dropout_prob,
                            norm=norm,
                            upsampling_mode=upsampling_mode))
            self.ups = nn.Sequential(*self.ups)

    def forward(self, input):
        return self.ups(input)

    def __init__(self):
        super().__init__()
        self.first_layers=[]
        self.first_layers.append(nn.Conv2d(1, 16, 1, bias=True, stride=1))
        self.first_layers=nn.Sequential(*self.first_layers)

        self.v1=[(DownBlock(in_channels=16,out_channels=32))]     #1024*2048
        self.v2=[(DownBlock(in_channels=32,out_channels=64))]     #512*1024
        self.v3=[(DownBlock(in_channels=64,out_channels=128))]    #256*512
        self.v4=[(DownBlock(in_channels=128,out_channels=256))]   #128*256
        self.v5=[(DownBlock(in_channels=256,out_channels=512))]   #64*128
        self.v6=[(DownBlock(in_channels=512,out_channels=1024))]  #32*64
        self.v7=[(DownBlock(in_channels=1024,out_channels=1024))] #16*32
        self.v8=[(DownBlock(in_channels=1024,out_channels=1024))] #8*16 ...
        self.v9=[(DownBlock(in_channels=1024,out_channels=1024))] #4*8->2*4 
        self.v10=[(DownBlock(in_channels=1024,out_channels=1024))] #2*4->1*2
        
        self.v1=nn.Sequential(*self.v1)
        self.v2=nn.Sequential(*self.v2)
        self.v3=nn.Sequential(*self.v3)
        self.v4=nn.Sequential(*self.v4)
        self.v5=nn.Sequential(*self.v5)
        self.v6=nn.Sequential(*self.v6)
        self.v7=nn.Sequential(*self.v7)
        self.v8=nn.Sequential(*self.v8)
        self.v9=nn.Sequential(*self.v9)
        self.v10=nn.Sequential(*self.v10)

        
        self.unet_first=[nn.Conv2d(2, 16, 1, bias=True, stride=1)]
        self.unet_first=nn.Sequential(*self.unet_first)

        self.u1=[DownBlock(in_channels=16,out_channels=32)]     #1024*2048 -> 512*1024
        self.u2=[DownBlock(in_channels=32,out_channels=64)]     #512*1024 -> 256*512
        self.u3=[DownBlock(in_channels=64,out_channels=128)]    #256*512 -> 128*256
        self.u4=[DownBlock(in_channels=128,out_channels=256)]   #128*256 -> 64*128
        self.u5=[DownBlock(in_channels=256,out_channels=512)]   #64*128 -> 32*64
        self.u6=[DownBlock(in_channels=512,out_channels=1024)]  #32*64 -> 16*32
        self.u7=[DownBlock(in_channels=1024,out_channels=1024)] #16*32 -> 8*16
        self.u8=[DownBlock(in_channels=1024,out_channels=1024)] #8*16 -> 4*8
        self.u9=[DownBlock(in_channels=1024,out_channels=1024)] #4*8 -> 2*4
        self.u10=[DownBlock(in_channels=1024,out_channels=1024)] #2*4 -> 1*2

        self.u1=nn.Sequential(*self.u1)
        self.u2=nn.Sequential(*self.u2)
        self.u3=nn.Sequential(*self.u3)
        self.u4=nn.Sequential(*self.u4)
        self.u5=nn.Sequential(*self.u5)
        self.u6=nn.Sequential(*self.u6)
        self.u7=nn.Sequential(*self.u7)
        self.u8=nn.Sequential(*self.u8)
        self.u9=nn.Sequential(*self.u9)
        self.u10=nn.Sequential(*self.u10)

        self.p1=[UpBlock(in_channels=2048,out_channels=1024)] #1*2 -> 2*4
        self.p2=[UpBlock(in_channels=3072,out_channels=1024)] #2*4 -> 4*8
        self.p3=[UpBlock(in_channels=3072,out_channels=1024)] #4*8 -> 8*16
        self.p4=[UpBlock(in_channels=3072,out_channels=1024)] #8*16 -> 16*32
        self.p5=[UpBlock(in_channels=3072,out_channels=512)] #16*32 -> 32*64
        self.p6=[UpBlock(in_channels=1536,out_channels=256)] #32*64 -> 64*128
        self.p7=[UpBlock(in_channels=768,out_channels=128)] #64*128 -> 128*256
        self.p8=[UpBlock(in_channels=384,out_channels=64)] #128*256 -> 256*512
        self.p9=[UpBlock(in_channels=192,out_channels=32)] #256*512 -> 512*1024
        self.p10=[UpBlock(in_channels=96,out_channels=16)] #512*1024 -> 1024*2048

        self.p1=nn.Sequential(*self.p1)
        self.p2=nn.Sequential(*self.p2)
        self.p3=nn.Sequential(*self.p3)
        self.p4=nn.Sequential(*self.p4)
        self.p5=nn.Sequential(*self.p5)
        self.p6=nn.Sequential(*self.p6)
        self.p7=nn.Sequential(*self.p7)
        self.p8=nn.Sequential(*self.p8)
        self.p9=nn.Sequential(*self.p9)
        self.p10=nn.Sequential(*self.p10)
        
        self.final_up=[nn.Conv2d(16,1, 1, bias=True, stride=1),
                       nn.Hardtanh(-math.pi,math.pi)]
        self.final_up=nn.Sequential(*self.final_up)

        
        
        
    
    def forward(self,input_img):
        #input_img[1] には 2チャンネルのペンギン(real,imag) 0130.png
        #input_img[0]には 1チャンネルのターゲット画像を入れてもらう


        vae_in=input_img[1]
        unet_in=input_img[0]

        r0=self.first_layers(vae_in)
        r1=self.v1(r0)      
        r2=self.v2(r1)     
        r3=self.v3(r2)
        r4=self.v4(r3)
        r5=self.v5(r4)
        r6=self.v6(r5)
        r7=self.v7(r6)
        r8=self.v8(r7)
        r9=self.v9(r8)
        r10=self.v10(r9)


        l0=self.unet_first(unet_in)
        l1=self.u1(l0)
        l2=self.u2(l1)
        l3=self.u3(l2)
        l4=self.u4(l3)
        l5=self.u5(l4)
        l6=self.u6(l5)
        l7=self.u7(l6)
        l8=self.u8(l7)
        l9=self.u9(l8)    #2*4*1024
        l10=self.u10(l9)  #1*2*1024

        under_cat=torch.cat([l10,r10],1) #1*2*2024
        l11=self.p1(under_cat) 
        second_cat=torch.cat([l11,l9,r9],1) #2*4*3072
        l12=self.p2(second_cat)
        l13=self.p3(torch.cat([l12,l8,r8],1))
        l14=self.p4(torch.cat([l13,l7,r7],1))
        l15=self.p5(torch.cat([l14,l6,r6],1))
        l16=self.p6(torch.cat([l15,l5,r5],1))
        l17=self.p7(torch.cat([l16,l4,r4],1))
        l18=self.p8(torch.cat([l17,l3,r3],1))
        l19=self.p9(torch.cat([l18,l2,r2],1))
        l20=self.p10(torch.cat([l19,l1,r1],1))

        res=self.final_up(l20)
                

        return res
    
class vunet_augumented_single(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layers=[]
        self.first_layers.append(nn.Conv2d(1, 16, 1, bias=True, stride=1))
        self.first_layers=nn.Sequential(*self.first_layers)


        self.v1=[(DownBlock(in_channels=16,out_channels=32))]     #1024*2048
        self.v2=[(DownBlock(in_channels=32,out_channels=64))]     #512*1024
        self.v3=[(DownBlock(in_channels=64,out_channels=128))]    #256*512
        self.v4=[(DownBlock(in_channels=128,out_channels=256))]   #128*256
        self.v5=[(DownBlock(in_channels=256,out_channels=512))]   #64*128
        self.v6=[(DownBlock(in_channels=512,out_channels=1024))]  #32*64
        self.v7=[(DownBlock(in_channels=1024,out_channels=1024))] #16*32
        self.v8=[(DownBlock(in_channels=1024,out_channels=1024))] #8*16 ...
        self.v9=[(DownBlock(in_channels=1024,out_channels=1024))] #4*8->2*4 
        self.v10=[(DownBlock(in_channels=1024,out_channels=1024))] #2*4->1*2
        
        self.v1=nn.Sequential(*self.v1)
        self.v2=nn.Sequential(*self.v2)
        self.v3=nn.Sequential(*self.v3)
        self.v4=nn.Sequential(*self.v4)
        self.v5=nn.Sequential(*self.v5)
        self.v6=nn.Sequential(*self.v6)
        self.v7=nn.Sequential(*self.v7)
        self.v8=nn.Sequential(*self.v8)
        self.v9=nn.Sequential(*self.v9)
        self.v10=nn.Sequential(*self.v10)

        
        self.unet_first=[nn.Conv2d(1, 16, 1, bias=True, stride=1)]
        self.unet_first=nn.Sequential(*self.unet_first)

        self.u1=[DownBlock(in_channels=16,out_channels=32)]     #1024*2048 -> 512*1024
        self.u2=[DownBlock(in_channels=32,out_channels=64)]     #512*1024 -> 256*512
        self.u3=[DownBlock(in_channels=64,out_channels=128)]    #256*512 -> 128*256
        self.u4=[DownBlock(in_channels=128,out_channels=256)]   #128*256 -> 64*128
        self.u5=[DownBlock(in_channels=256,out_channels=512)]   #64*128 -> 32*64
        self.u6=[DownBlock(in_channels=512,out_channels=1024)]  #32*64 -> 16*32
        self.u7=[DownBlock(in_channels=1024,out_channels=1024)] #16*32 -> 8*16
        self.u8=[DownBlock(in_channels=1024,out_channels=1024)] #8*16 -> 4*8
        self.u9=[DownBlock(in_channels=1024,out_channels=1024)] #4*8 -> 2*4
        self.u10=[DownBlock(in_channels=1024,out_channels=1024)] #2*4 -> 1*2

        self.u1=nn.Sequential(*self.u1)
        self.u2=nn.Sequential(*self.u2)
        self.u3=nn.Sequential(*self.u3)
        self.u4=nn.Sequential(*self.u4)
        self.u5=nn.Sequential(*self.u5)
        self.u6=nn.Sequential(*self.u6)
        self.u7=nn.Sequential(*self.u7)
        self.u8=nn.Sequential(*self.u8)
        self.u9=nn.Sequential(*self.u9)
        self.u10=nn.Sequential(*self.u10)

        self.p1=[UpBlock(in_channels=2048,out_channels=1024)] #1*2 -> 2*4
        self.p2=[UpBlock(in_channels=3072,out_channels=1024)] #2*4 -> 4*8
        self.p3=[UpBlock(in_channels=3072,out_channels=1024)] #4*8 -> 8*16
        self.p4=[UpBlock(in_channels=3072,out_channels=1024)] #8*16 -> 16*32
        self.p5=[UpBlock(in_channels=3072,out_channels=512)] #16*32 -> 32*64
        self.p6=[UpBlock(in_channels=1536,out_channels=256)] #32*64 -> 64*128
        self.p7=[UpBlock(in_channels=768,out_channels=128)] #64*128 -> 128*256
        self.p8=[UpBlock(in_channels=384,out_channels=64)] #128*256 -> 256*512
        self.p9=[UpBlock(in_channels=192,out_channels=32)] #256*512 -> 512*1024
        self.p10=[UpBlock(in_channels=96,out_channels=16)] #512*1024 -> 1024*2048

        self.p1=nn.Sequential(*self.p1)
        self.p2=nn.Sequential(*self.p2)
        self.p3=nn.Sequential(*self.p3)
        self.p4=nn.Sequential(*self.p4)
        self.p5=nn.Sequential(*self.p5)
        self.p6=nn.Sequential(*self.p6)
        self.p7=nn.Sequential(*self.p7)
        self.p8=nn.Sequential(*self.p8)
        self.p9=nn.Sequential(*self.p9)
        self.p10=nn.Sequential(*self.p10)
        
        self.final_up=[nn.Conv2d(16,1, 1, bias=True, stride=1),
                       nn.Hardtanh(-math.pi,math.pi)]
        self.final_up=nn.Sequential(*self.final_up)

        
        
        
    
    def forward(self,input_img):
        #input_img[1] には 1チャンネルのinput
        #input_img[0]には 1チャンネルのターゲット画像を入れてもらう

        vae_in=input_img[1]
        unet_in=input_img[0]

        r0=self.first_layers(vae_in)
        r1=self.v1(r0)      
        r2=self.v2(r1)     
        r3=self.v3(r2)
        r4=self.v4(r3)
        r5=self.v5(r4)
        r6=self.v6(r5)
        r7=self.v7(r6)
        r8=self.v8(r7)
        r9=self.v9(r8)
        r10=self.v10(r9)


        l0=self.unet_first(unet_in)
        l1=self.u1(l0)
        l2=self.u2(l1)
        l3=self.u3(l2)
        l4=self.u4(l3)
        l5=self.u5(l4)
        l6=self.u6(l5)
        l7=self.u7(l6)
        l8=self.u8(l7)
        l9=self.u9(l8)    #2*4*1024
        l10=self.u10(l9)  #1*2*1024

        under_cat=torch.cat([l10,r10],1) #1*2*2024
        l11=self.p1(under_cat) 
        second_cat=torch.cat([l11,l9,r9],1) #2*4*3072
        l12=self.p2(second_cat)
        l13=self.p3(torch.cat([l12,l8,r8],1))
        l14=self.p4(torch.cat([l13,l7,r7],1))
        l15=self.p5(torch.cat([l14,l6,r6],1))
        l16=self.p6(torch.cat([l15,l5,r5],1))
        l17=self.p7(torch.cat([l16,l4,r4],1))
        l18=self.p8(torch.cat([l17,l3,r3],1))
        l19=self.p9(torch.cat([l18,l2,r2],1))
        l20=self.p10(torch.cat([l19,l1,r1],1))

        res=self.final_up(l20)
                

        return res
        
class vunet_augumented(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layers=[]
        self.first_layers.append(nn.Conv2d(1, 16, 1, bias=True, stride=1))
        self.first_layers=nn.Sequential(*self.first_layers)


        self.v1=[(DownBlock(in_channels=16,out_channels=32))]     #1024*2048
        self.v2=[(DownBlock(in_channels=32,out_channels=64))]     #512*1024
        self.v3=[(DownBlock(in_channels=64,out_channels=128))]    #256*512
        self.v4=[(DownBlock(in_channels=128,out_channels=256))]   #128*256
        self.v5=[(DownBlock(in_channels=256,out_channels=512))]   #64*128
        self.v6=[(DownBlock(in_channels=512,out_channels=1024))]  #32*64
        self.v7=[(DownBlock(in_channels=1024,out_channels=1024))] #16*32
        self.v8=[(DownBlock(in_channels=1024,out_channels=1024))] #8*16 ...
        self.v9=[(DownBlock(in_channels=1024,out_channels=1024))] #4*8->2*4 
        self.v10=[(DownBlock(in_channels=1024,out_channels=1024))] #2*4->1*2
        
        self.v1=nn.Sequential(*self.v1)
        self.v2=nn.Sequential(*self.v2)
        self.v3=nn.Sequential(*self.v3)
        self.v4=nn.Sequential(*self.v4)
        self.v5=nn.Sequential(*self.v5)
        self.v6=nn.Sequential(*self.v6)
        self.v7=nn.Sequential(*self.v7)
        self.v8=nn.Sequential(*self.v8)
        self.v9=nn.Sequential(*self.v9)
        self.v10=nn.Sequential(*self.v10)

        
        self.unet_first=[nn.Conv2d(2, 16, 1, bias=True, stride=1)]
        self.unet_first=nn.Sequential(*self.unet_first)

        self.u1=[DownBlock(in_channels=16,out_channels=32)]     #1024*2048 -> 512*1024
        self.u2=[DownBlock(in_channels=32,out_channels=64)]     #512*1024 -> 256*512
        self.u3=[DownBlock(in_channels=64,out_channels=128)]    #256*512 -> 128*256
        self.u4=[DownBlock(in_channels=128,out_channels=256)]   #128*256 -> 64*128
        self.u5=[DownBlock(in_channels=256,out_channels=512)]   #64*128 -> 32*64
        self.u6=[DownBlock(in_channels=512,out_channels=1024)]  #32*64 -> 16*32
        self.u7=[DownBlock(in_channels=1024,out_channels=1024)] #16*32 -> 8*16
        self.u8=[DownBlock(in_channels=1024,out_channels=1024)] #8*16 -> 4*8
        self.u9=[DownBlock(in_channels=1024,out_channels=1024)] #4*8 -> 2*4
        self.u10=[DownBlock(in_channels=1024,out_channels=1024)] #2*4 -> 1*2

        self.u1=nn.Sequential(*self.u1)
        self.u2=nn.Sequential(*self.u2)
        self.u3=nn.Sequential(*self.u3)
        self.u4=nn.Sequential(*self.u4)
        self.u5=nn.Sequential(*self.u5)
        self.u6=nn.Sequential(*self.u6)
        self.u7=nn.Sequential(*self.u7)
        self.u8=nn.Sequential(*self.u8)
        self.u9=nn.Sequential(*self.u9)
        self.u10=nn.Sequential(*self.u10)

        self.p1=[UpBlock(in_channels=2048,out_channels=1024)] #1*2 -> 2*4
        self.p2=[UpBlock(in_channels=3072,out_channels=1024)] #2*4 -> 4*8
        self.p3=[UpBlock(in_channels=3072,out_channels=1024)] #4*8 -> 8*16
        self.p4=[UpBlock(in_channels=3072,out_channels=1024)] #8*16 -> 16*32
        self.p5=[UpBlock(in_channels=3072,out_channels=512)] #16*32 -> 32*64
        self.p6=[UpBlock(in_channels=1536,out_channels=256)] #32*64 -> 64*128
        self.p7=[UpBlock(in_channels=768,out_channels=128)] #64*128 -> 128*256
        self.p8=[UpBlock(in_channels=384,out_channels=64)] #128*256 -> 256*512
        self.p9=[UpBlock(in_channels=192,out_channels=32)] #256*512 -> 512*1024
        self.p10=[UpBlock(in_channels=96,out_channels=16)] #512*1024 -> 1024*2048

        self.p1=nn.Sequential(*self.p1)
        self.p2=nn.Sequential(*self.p2)
        self.p3=nn.Sequential(*self.p3)
        self.p4=nn.Sequential(*self.p4)
        self.p5=nn.Sequential(*self.p5)
        self.p6=nn.Sequential(*self.p6)
        self.p7=nn.Sequential(*self.p7)
        self.p8=nn.Sequential(*self.p8)
        self.p9=nn.Sequential(*self.p9)
        self.p10=nn.Sequential(*self.p10)
        
        self.final_up=[nn.Conv2d(16,1, 1, bias=True, stride=1),
                       nn.Hardtanh(-math.pi,math.pi)]
        self.final_up=nn.Sequential(*self.final_up)

        
        
        
    
    def forward(self,input_img):
        #input_img[0] には 2チャンネルのペンギン(real,imag) 0130.png
        #input_img[1]には 1チャンネルのターゲット画像を入れてもらう


        vae_in=input_img[0]
        unet_in=input_img[1]

        r0=self.first_layers(vae_in)
        r1=self.v1(r0)      
        r2=self.v2(r1)     
        r3=self.v3(r2)
        r4=self.v4(r3)
        r5=self.v5(r4)
        r6=self.v6(r5)
        r7=self.v7(r6)
        r8=self.v8(r7)
        r9=self.v9(r8)
        r10=self.v10(r9)


        l0=self.unet_first(unet_in)
        l1=self.u1(l0)
        l2=self.u2(l1)
        l3=self.u3(l2)
        l4=self.u4(l3)
        l5=self.u5(l4)
        l6=self.u6(l5)
        l7=self.u7(l6)
        l8=self.u8(l7)
        l9=self.u9(l8)    #2*4*1024
        l10=self.u10(l9)  #1*2*1024

        under_cat=torch.cat([l10,r10],1) #1*2*2024
        l11=self.p1(under_cat) 
        second_cat=torch.cat([l11,l9,r9],1) #2*4*3072
        l12=self.p2(second_cat)
        l13=self.p3(torch.cat([l12,l8,r8],1))
        l14=self.p4(torch.cat([l13,l7,r7],1))
        l15=self.p5(torch.cat([l14,l6,r6],1))
        l16=self.p6(torch.cat([l15,l5,r5],1))
        l17=self.p7(torch.cat([l16,l4,r4],1))
        l18=self.p8(torch.cat([l17,l3,r3],1))
        l19=self.p9(torch.cat([l18,l2,r2],1))
        l20=self.p10(torch.cat([l19,l1,r1],1))

        res=self.final_up(l20)
                

        return res


class vunet_augumented_original(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layers=[]
        self.first_layers.append(nn.Conv2d(1, 16, 1, bias=True, stride=1))
        self.first_layers=nn.Sequential(*self.first_layers)


        self.v1=[(DownBlock(in_channels=16,out_channels=32))]     #1024*2048
        self.v2=[(DownBlock(in_channels=32,out_channels=64))]     #512*1024
        self.v3=[(DownBlock(in_channels=64,out_channels=128))]    #256*512
        self.v4=[(DownBlock(in_channels=128,out_channels=256))]   #128*256
        self.v5=[(DownBlock(in_channels=256,out_channels=512))]   #64*128
        self.v6=[(DownBlock(in_channels=512,out_channels=1024))]  #32*64
        self.v7=[(DownBlock(in_channels=1024,out_channels=1024))] #16*32
        self.v8=[(DownBlock(in_channels=1024,out_channels=1024))] #8*16 ...
        self.v9=[(DownBlock(in_channels=1024,out_channels=1024))] #4*8->2*4 
        self.v10=[(DownBlock(in_channels=1024,out_channels=1024))] #2*4->1*2
        
        self.v1=nn.Sequential(*self.v1)
        self.v2=nn.Sequential(*self.v2)
        self.v3=nn.Sequential(*self.v3)
        self.v4=nn.Sequential(*self.v4)
        self.v5=nn.Sequential(*self.v5)
        self.v6=nn.Sequential(*self.v6)
        self.v7=nn.Sequential(*self.v7)
        self.v8=nn.Sequential(*self.v8)
        self.v9=nn.Sequential(*self.v9)
        self.v10=nn.Sequential(*self.v10)

        
        self.unet_first=[nn.Conv2d(1, 16, 1, bias=True, stride=1)]
        self.unet_first=nn.Sequential(*self.unet_first)

        self.u1=[DownBlock(in_channels=16,out_channels=32)]     #1024*2048 -> 512*1024
        self.u2=[DownBlock(in_channels=32,out_channels=64)]     #512*1024 -> 256*512
        self.u3=[DownBlock(in_channels=64,out_channels=128)]    #256*512 -> 128*256
        self.u4=[DownBlock(in_channels=128,out_channels=256)]   #128*256 -> 64*128
        self.u5=[DownBlock(in_channels=256,out_channels=512)]   #64*128 -> 32*64
        self.u6=[DownBlock(in_channels=512,out_channels=1024)]  #32*64 -> 16*32
        self.u7=[DownBlock(in_channels=1024,out_channels=1024)] #16*32 -> 8*16
        self.u8=[DownBlock(in_channels=1024,out_channels=1024)] #8*16 -> 4*8
        self.u9=[DownBlock(in_channels=1024,out_channels=1024)] #4*8 -> 2*4
        self.u10=[DownBlock(in_channels=1024,out_channels=1024)] #2*4 -> 1*2

        self.u1=nn.Sequential(*self.u1)
        self.u2=nn.Sequential(*self.u2)
        self.u3=nn.Sequential(*self.u3)
        self.u4=nn.Sequential(*self.u4)
        self.u5=nn.Sequential(*self.u5)
        self.u6=nn.Sequential(*self.u6)
        self.u7=nn.Sequential(*self.u7)
        self.u8=nn.Sequential(*self.u8)
        self.u9=nn.Sequential(*self.u9)
        self.u10=nn.Sequential(*self.u10)

        self.p1=[UpBlock(in_channels=2048,out_channels=1024)] #1*2 -> 2*4
        self.p2=[UpBlock(in_channels=3072,out_channels=1024)] #2*4 -> 4*8
        self.p3=[UpBlock(in_channels=3072,out_channels=1024)] #4*8 -> 8*16
        self.p4=[UpBlock(in_channels=3072,out_channels=1024)] #8*16 -> 16*32
        self.p5=[UpBlock(in_channels=3072,out_channels=512)] #16*32 -> 32*64
        self.p6=[UpBlock(in_channels=1536,out_channels=256)] #32*64 -> 64*128
        self.p7=[UpBlock(in_channels=768,out_channels=128)] #64*128 -> 128*256
        self.p8=[UpBlock(in_channels=384,out_channels=64)] #128*256 -> 256*512
        self.p9=[UpBlock(in_channels=192,out_channels=32)] #256*512 -> 512*1024
        self.p10=[UpBlock(in_channels=96,out_channels=16)] #512*1024 -> 1024*2048

        self.p1=nn.Sequential(*self.p1)
        self.p2=nn.Sequential(*self.p2)
        self.p3=nn.Sequential(*self.p3)
        self.p4=nn.Sequential(*self.p4)
        self.p5=nn.Sequential(*self.p5)
        self.p6=nn.Sequential(*self.p6)
        self.p7=nn.Sequential(*self.p7)
        self.p8=nn.Sequential(*self.p8)
        self.p9=nn.Sequential(*self.p9)
        self.p10=nn.Sequential(*self.p10)
        
        self.final_up=[nn.Conv2d(16,1, 1, bias=True, stride=1),
                       nn.Hardtanh(-math.pi,math.pi)]
        self.final_up=nn.Sequential(*self.final_up)

        
        
        
    
    def forward(self,input_img):
        #input_img[1] には 1チャンネルのinput
        #input_img[0]には 1チャンネルのターゲット画像を入れてもらう

        vae_in=input_img[1]
        unet_in=input_img[0]

        r0=self.first_layers(vae_in)
        r1=self.v1(r0)      
        r2=self.v2(r1)     
        r3=self.v3(r2)
        r4=self.v4(r3)
        r5=self.v5(r4)
        r6=self.v6(r5)
        r7=self.v7(r6)
        r8=self.v8(r7)
        r9=self.v9(r8)
        r10=self.v10(r9)


        l0=self.unet_first(unet_in)
        l1=self.u1(l0)
        l2=self.u2(l1)
        l3=self.u3(l2)
        l4=self.u4(l3)
        l5=self.u5(l4)
        l6=self.u6(l5)
        l7=self.u7(l6)
        l8=self.u8(l7)
        l9=self.u9(l8)    #2*4*1024
        l10=self.u10(l9)  #1*2*1024

        under_cat=torch.cat([l10,r10],1) #1*2*2024
        l11=self.p1(under_cat) 
        second_cat=torch.cat([l11,l9,r9],1) #2*4*3072
        l12=self.p2(second_cat)
        l13=self.p3(torch.cat([l12,l8,r8],1))
        l14=self.p4(torch.cat([l13,l7,r7],1))
        l15=self.p5(torch.cat([l14,l6,r6],1))
        l16=self.p6(torch.cat([l15,l5,r5],1))
        l17=self.p7(torch.cat([l16,l4,r4],1))
        l18=self.p8(torch.cat([l17,l3,r3],1))
        l19=self.p9(torch.cat([l18,l2,r2],1))
        l20=self.p10(torch.cat([l19,l1,r1],1))

        res=self.final_up(l20)
                

        return res