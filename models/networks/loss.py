import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util
from .stylegan2_layers import Downsample


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)
