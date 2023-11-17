import csv
import time
import os
import random
import cmath
import sys
import imageio
import os
import skimage.io
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import configargparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import utils.utils as utils
# from utils.distanced_image_loader import DistnaceImageLoader
# from utils.original_image_loader import OriginalImageLoader
from utils.augmented_image_loader import ImageLoader
from utils.kernel_loader import KernelLoader
from propagation_model import ModelPropagate
from utils.modules import DPAC, SGD, GS
from holonet import *
from propagation_ASM import propagation_ASM

phase_filename = "hoge"
slm_phase = skimage.io.imread(phase_filename) / 255.
slm_phase = torch.tensor((1 - slm_phase) * 2 * np.pi - np.pi, dtype=dtype).reshape(1, 1, *slm_res).to(device)

# propagate field
real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
slm_field = torch.complex(real, imag)

if opt.prop_model.upper() == 'MODEL':
    propagator = propagators[c]  # Select CITL-calibrated models for each channel
recon_field = utils.propagate_field(slm_field, propagator, prop_dists[c], wavelengths[c], feature_size,
                                    opt.prop_model, dtype)

# cartesian to polar coordinate
recon_amp_c = recon_field.abs()

# crop to ROI
recon_amp_c = utils.crop_image(recon_amp_c, target_shape=roi_res, stacked_complex=False)

# append to list
recon_amp.append(recon_amp_c)

# list to tensor, scaling
recon_amp = torch.cat(recon_amp, dim=1)
recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))

# tensor to numpy
recon_amp = recon_amp.squeeze().cpu().detach().numpy()



# save reconstructed image in srgb domain
recon_srgb = utils.srgb_lin2gamma(np.clip(recon_amp**2, 0.0, 1.0))
utils.cond_mkdir(recon_path)
imageio.imwrite(os.path.join(recon_path, f'{target_idx}_{run_id}_{chan_strs[channel]}.png'), (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))
