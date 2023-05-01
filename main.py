"""
Conditional Neural holography:

This code reproduces the experiments for Conditional Neural Holography. 
A majority of this code utilizes Neural Holography by Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein 2020.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.


@article{Peng:2020:NeuralHolography,
author = {Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein},
title = {{Neural Holography with Camera-in-the-loop Training}},
journal = {ACM Trans. Graph. (SIGGRAPH Asia)},
year = {2020},
}

-----

"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import configargparse
from torch.utils.tensorboard import SummaryWriter

import utils.utils as utils
from utils.distanced_image_loader import DistnaceImageLoader
from utils.original_image_loader import OriginalImageLoader
from propagation_model import ModelPropagate
# from utils.modules import SGD, GS, DPAC, PhysicalProp
from holonet import *
from propagation_ASM import propagation_ASM

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--train', type=bool, default=True, help='train:True, evaluate:False')
p.add_argument('--model_type', type=bool, default=True, help='Augmented Holonet:True, Augmented Conditional Unet:False')
p.add_argument('--distance_to_image', type=bool, default=True, help='Zone Plate:True, Reflect Changed Phase:False')
p.add_argument('--compare', type=bool, default=False, help='if True, this code will compare to DPAC and back propagation')
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='./images/div_and_flickr', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
               help='Directory for the pretrained holonet/unet network')

# parse arguments
opt = p.parse_args()
TRAIN=opt.train
MODEL_TYPE=opt.model_type
DISTANCE_TO_IMAGE=opt.distance_to_image
status_name="Train" if TRAIN else "Eavl"
model_type_name="Augmented_Holonet" if MODEL_TYPE else "Augmented Conditional Unet"
distance_to_image_name="Zone_Plate" if DISTANCE_TO_IMAGE else "Reflect Changed Phase"
run_id = f"{status_name}_{model_type_name}_{distance_to_image_name}"
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
print(f'   - optimizing phase with {model_type_name}/{distance_to_image_name} ... ')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1080, 1920)  # resolution of SLM
image_res = (1080, 1920)
roi_res = (880, 1600)  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 0.95  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

propagator = propagation_ASM  # Ideal model
Augmented_Holonet=HoloZonePlateNet(  
        wavelength=wavelength,
        feature_size=feature_size[0],
        initial_phase=InitialDoubleUnet(6, 16),
        final_phase_only=FinalPhaseOnlyUnet(8, 32, num_in=2)
)
Augmented_Conditional_Unet=VUnet_Aug_single()
phase_only_algorithm = Augmented_Holonet if MODEL_TYPE else Augmented_Conditional_Unet

# elif opt.method == 'DPAC':
#     phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)
image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)





# Loop over the dataset
for k, target in enumerate(image_loader):
    # get target image
    target_amp, target_res, target_filename = target
    target_path, target_filename = os.path.split(target_filename[0])
    target_idx = target_filename.split('_')[-1]
    target_amp = target_amp.to(device)
    print(target_idx)

    # if you want to separate folders by target_idx or whatever, you can do so here.
    phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
    phase_only_algorithm.phase_path = os.path.join(root_path)

    # run algorithm (See algorithm_modules.py and algorithms.py)
    if opt.method in ['DPAC', 'HOLONET', 'UNET']:
        # direct methods
        _, final_phase = phase_only_algorithm(target_amp)
    else:
        # iterative methods, initial phase: random guess
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
        final_phase = phase_only_algorithm(target_amp, init_phase)

    print(final_phase.shape)

    # save the final result somewhere.
    phase_out_8bit = utils.phasemap_8bit(final_phase.cpu().detach(), inverted=True)

    utils.cond_mkdir(root_path)
    cv2.imwrite(os.path.join(root_path, f'{target_idx}.png'), phase_out_8bit)

print(f'    - Done, result: --root_path={root_path}')
