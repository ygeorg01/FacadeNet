import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import util
from models.networks import BaseNetwork
from models.networks.stylegan2_layers import ResBlock, ConvLayer, ToRGB, EqualLinear, Blur, Upsample, make_kernel
from models.networks.stylegan2_op import upfirdn2d


class ToSpatialCode(torch.nn.Module):
    def __init__(self, inch, outch, scale):
        super().__init__()
        hiddench = inch // 2
        # self.conv1 = ConvLayer(inch, hiddench, 1, activate=True, bias=True, reflection_pad=True)
        # self.conv2 = ConvLayer(hiddench, outch, 1, activate=False, bias=True, reflection_pad=True)
        self.conv1 = ConvLayer(inch, hiddench, 1, activate=True, bias=True)
        self.conv2 = ConvLayer(hiddench, outch, 1, activate=False, bias=True)
        self.scale = scale
        self.upsample = Upsample([1, 3, 3, 1], 2)
        self.blur = Blur([1, 3, 3, 1], pad=(2, 1))
        self.register_buffer('kernel', make_kernel([1, 3, 3, 1]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(int(np.log2(self.scale))):
            x = self.upsample(x)
        return x


class StyleGAN2ResnetEncoderNostyle(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
        parser.add_argument("--netE_num_downsampling_sp", default=5, type=int)
        parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
        parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(3, self.nc(0), 1, reflection_pad=True))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_sp):

            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel, reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",

            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True, reflection_pad=True),
                ConvLayer(nchannels, self.opt.spatial_code_ch, kernel_size=1,
                      activate=False, bias=True, reflection_pad=True)
            )
        )



    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        # nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)

        sp = util.normalize(sp)

        return sp