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
import random
import cmath
import sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import configargparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import utils.utils as utils

# from utils.distanced_image_loader import DistnaceImageLoader
# from utils.original_image_loader import OriginalImageLoader

from utils.augmented_image_loader import ImageLoader
from utils.kernel_loader import KernelLoader
from propagation_model import ModelPropagate
# from utils.modules import SGD, GS, DPAC, PhysicalProp
from holonet import *
from propagation_ASM import propagation_ASM

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--train', type=int, default=0, help='train:0, evaluate:1')
p.add_argument('--model_type', type=int, default=0, help='Augmented Holonet:0, Augmented Conditional Unet:1')
p.add_argument('--distance_to_image', type=int, default=0, help='Zone Plate:0, Reflect Changed Phase:1')
p.add_argument('--compare', type=int, default=1, help='if 0, this code will compare to DPAC and back propagation')
p.add_argument('--root_path', type=str, default='/images/phases', help='Directory where optimized phases will be saved.')
p.add_argument('--kernel_path', type=str, default='/images/kernels', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='/images/div_and_flickr', help='Directory for the dataset')
p.add_argument('--val_path', type=str, default='/images/DIV2K_valid_HR', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
               help='Directory for the pretrained holonet/unet network')
p.add_argument('--start_dis', type=float, default=0.2, help='z_0[m]')
p.add_argument('--alpha', type=float, default=2.0, help='phase_shift')
p.add_argument('--end_dis', type=int, default=200000, help='end of distances')
p.add_argument('--num_split', type=int, default=100, help='number of distance points')
p.add_argument('--plate_path', type=str, default='/images/zoneplates', help='Directory where optimized phases will be saved.')
p.add_argument('--original',type=int, default=0,help="if use Original HoloNet, set 1")


# parse arguments
opt = p.parse_args()
TRAIN= opt.train==0
MODEL_TYPE=opt.model_type==0
DISTANCE_TO_IMAGE= opt.distance_to_image==0 
ORIGINAL=opt.original==1
status_name="Train" if TRAIN else "Eavl"
model_type_name="Augmented_Holonet" if MODEL_TYPE else "Augmented Conditional Unet"
if ORIGINAL:
    model_type_name="Original"
distance_to_image_name="Zone_Plate" if DISTANCE_TO_IMAGE else "Reflect Changed Phase"
start_dis=opt.start_dis
alpha=opt.alpha
num_splits=opt.num_split
end_dis=opt.end_dis
run_id = f"{status_name}_{model_type_name}_{distance_to_image_name}_{start_dis}_{end_dis}_{num_splits}"
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
print(f'   - optimizing phase with {run_id} ... ')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1024, 2048)  # resolution of SLM
image_res=roi_res=slm_res
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using

# Options for the algorithm
loss = nn.MSELoss().to(device)  # loss functions to use (try other loss functions!)
s0 = 0.95  # initial scale

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases
kernel_path=os.path.join(opt.kernel_path,f'{start_dis}_{end_dis}_{num_splits}')
plate_path=os.path.join(opt.plate_path,f'{start_dis}_{end_dis}_{num_splits}')


# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)

### Distance
start = start_dis+wavelength/alpha
end = start_dis+wavelength/alpha+end_dis*wavelength

step = (end - start) / num_splits
distancebox = [start + step * i for i in range(num_splits + 1)]

if ORIGINAL:
    distancebox=[start]


if(os.path.isdir(kernel_path)):
    pass
else:
    os.mkdir(kernel_path)
    for c,d in enumerate(distancebox):
        with torch.no_grad():
            temp_H=propagation_ASM(torch.empty([1,1,slm_res[0],slm_res[1]], dtype=torch.complex64), feature_size,wavelength,d, return_H=True)
            torch.save(temp_H,f'{kernel_path}/{c}.pth')
            print(c,temp_H)
            del temp_H
            print(f"Calculating Kernel {c+1}/{len(distancebox)}")
if(os.path.isdir(f'{kernel_path}_back')):
    pass
else:
    os.mkdir(f'{kernel_path}_back')
    for c,d in enumerate(distancebox):
        with torch.no_grad():
            temp_H=propagation_ASM(torch.empty([1,1,slm_res[0],slm_res[1]], dtype=torch.complex64), feature_size,wavelength,-d, return_H=True)
            torch.save(temp_H,f'{kernel_path}_back/{c}.pth')
            del temp_H
            print(f"Calculating Kernel Back {c+1}/{len(distancebox)}")

init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
kLoader=KernelLoader(kernel_path)
kbLoder=KernelLoader(f'{kernel_path}_back')

if(os.path.isdir(plate_path)):
    pass
else:
    os.mkdir(plate_path)
    for c,d in enumerate(distancebox):
        with torch.no_grad():
            
            point_light=torch.zeros([1,1,slm_res[0],slm_res[1]],dtype=torch.float32).to(device)
            point_light[0, 0, slm_res[0]//2-1:slm_res[0]//2+1, slm_res[1]//2-1:slm_res[1]//2+1] = 1.0
            zone_comp= propagation_ASM(point_light, feature_size,
                                wavelength, -d,
                                precomped_H=kbLoder[c].to(device),
                                linear_conv=True)
            zone_plate=torch.angle(zone_comp)
            zone_plate=(zone_plate-torch.min(zone_plate))/(torch.max(zone_plate)-torch.min(zone_plate))        
            torch.save(zone_plate,f'{plate_path}/{c}.pth')
            del zone_plate
            print(f"Calculating Zone Plate {c+1}/{len(distancebox)}")

plateLoader=KernelLoader(plate_path)

### 

propagator = propagation_ASM  # Ideal model
Augmented_Holonet=HoloZonePlateNet(  
        wavelength=wavelength,
        feature_size=feature_size[0],
        initial_phase=InitialDoubleUnet(6, 16),
        final_phase_only=FinalPhaseOnlyUnet(8, 32, num_in=2),
        distace_box=distancebox,
        target_shape=[1,1,slm_res[0],slm_res[1]]
) if DISTANCE_TO_IMAGE else HoloZonePlateNet2ch(
        wavelength=wavelength,
        feature_size=feature_size[0],
        initial_phase=InitialDoubleUnet(6, 16),
        final_phase_only=FinalPhaseOnlyUnet(8, 32, num_in=2),
        distance_box=distancebox,
        target_shape=[1,1,slm_res[0],slm_res[1]]
)

Augmented_Conditional_Unet=VUnet_Aug_single(  
        target_shpae=[1,1,slm_res[0],slm_res[1]],
        feature_size=feature_size,
        wavelength=wavelength,
        distance_box=distancebox,) if DISTANCE_TO_IMAGE else VUnet_Aug(target_shpae=[1,1,slm_res[0],slm_res[1]],
        feature_size=feature_size,
        wavelength=wavelength,
        distance_box=distancebox)

phase_generator = Augmented_Holonet if MODEL_TYPE else Augmented_Conditional_Unet
if ORIGINAL:
    phase_generator = HoloNet(
        distance=start,
        wavelength=wavelength,
        initial_phase=InitialPhaseUnet(4, 16),
        final_phase_only=FinalPhaseOnlyUnet(4, 16, num_in=2))
phase_generator.to(device)
phase_generator.train()  # generator to be trained
optvars = phase_generator.parameters()

# elif opt.method == 'DPAC':
#     phase_only_algorithm = DPAC(prop_dist, wavelength, feature_size, opt.prop_model, propagator, device)

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)

###################

mse_loss = nn.MSELoss()

# upload to GPU
loss = loss.to(device)
mse_loss = mse_loss.to(device)

# create optimizer
# if warm_start:
#     opt.lr /= 10
optimizer = optim.Adam(optvars, lr=1e-4)

image_loader = ImageLoader(opt.data_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

# loads images from disk, set to augment with flipping
val_loader =  ImageLoader(opt.val_path, channel=channel,
                           image_res=image_res, homography_res=roi_res,
                           crop_to_homography=True,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

num_mse_iters = 500
num_mse_epochs=0
num_epochs=10
#################
# Training loop #
#################

ik=0


for i in range(num_epochs):    
    for k, target in enumerate(image_loader):

        # get target image
        target_amp, target_res,target_filename = target

        dis_k=0 if ORIGINAL else random.randrange(num_splits)
        

        preH=kLoader[dis_k].to(device)
        preHb=kbLoder[dis_k].to(device)
        plate=plateLoader[dis_k].to(device)
   

        ik+=1
        target_amp = target_amp.to(device)
        # distance=distance[0]
        optimizer.zero_grad()
        # forward model
        if ORIGINAL:
            _,slm_phase=phase_generator(target_amp)
        else:
            slm_phase=phase_generator(target_amp,plate,dis_k,preHb)


        real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(real, imag)
        output_complex=utils.propagate_field(slm_field,propagator,distancebox[dis_k],wavelength,feature_size,"ASM",dtype = torch.float32,precomputed_H=preH)

        output_lin_intensity = torch.sum(output_complex.abs()**2 * 0.95, dim=1, keepdim=True)
        output_amp = torch.pow(output_lin_intensity, 0.5)

        target_res=target_res[0]

        # crop outputs to the region we care about
        # output_amp = utils.crop_image(output_amp, target_res, stacked_complex=False)         
        # target_amp = utils.crop_image(target_amp, target_res, stacked_complex=False)
        with torch.no_grad():
            scaled_out = output_amp * target_amp.mean() / output_amp.mean()
        output_amp = output_amp + (scaled_out - output_amp).detach()        

        loss_main=loss(output_amp,target_amp)

        loss_main.backward()
        optimizer.step()

        ## PSNRを計算

        def psnr(tensor1, tensor2):
            mse = F.mse_loss(tensor1, tensor2)
            max_pixel_value = 1
            psnr = 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())
            return psnr
        
        # GPU上のテンソルをCPUにコピー
        target_amp_cpu = target_amp[0,0,:,:].to('cpu')
        loss_main_cpu =output_amp[0,0,:,:].to('cpu')

        # 画像として見做した時のPSNRを計算
        psnr_value = psnr(target_amp_cpu, loss_main_cpu)

        print(f'iteration {ik}:Loss:{loss_main.item()} PSNR:{psnr_value}w/{target_filename}@{distancebox[dis_k]}')

        with torch.no_grad():
            writer.add_scalar('Loss', loss_main, ik)
        
            if ik % 50 == 0:

                # write images and loss to tensorboard
                writer.add_image('Target Amplitude', target_amp[0, ...], ik)

                # normalize reconstructed amplitude
                output_amp0 = output_amp[0, ...]
                maxVal = torch.max(output_amp0)
                minVal = torch.min(output_amp0)
                tmp = (output_amp0 - minVal) / (maxVal - minVal)
                writer.add_image('Reconstruction Amplitude', tmp, ik)

                # normalize SLM phase
                writer.add_image('SLM Phase', (slm_phase[0, ...] + math.pi) / (2 * math.pi), ik)

                for m,val_target in enumerate(val_loader):
                    if m==0:

                        
                        phase_generator.eval()
                        # _,target_amp, target_res, distance,target_filename = target
                        val_amp, val_res,_ = val_target
                        
                        val_amp=val_amp.to(device)
                        val_k=0 if ORIGINAL else random.randrange(num_splits)
                        preH=kLoader[val_k].to(device)
                        preHb=kbLoder[val_k].to(device)
                        val_plate=plateLoader[val_k].to(device)
                        if ORIGINAL:
                            _,val_phase=phase_generator(val_amp)
                        else:
                            val_phase=phase_generator(val_amp,val_plate,val_k,preHb)

                        real, imag = utils.polar_to_rect(torch.ones_like(val_phase),val_phase)
                        slm_field = torch.complex(real, imag)
                        output_complex_val=utils.propagate_field(slm_field,propagator,distancebox[val_k],wavelength,feature_size,"ASM",dtype = torch.float32,precomputed_H=preH)

                        output_lin_intensity_val = torch.sum(output_complex_val.abs()**2 * 0.95, dim=1, keepdim=True)

                        output_amp_val = torch.pow(output_lin_intensity_val, 0.5)

                        scaled_out = output_amp_val * val_amp.mean() / output_amp_val.mean()
                        output_amp_val= output_amp_val + (scaled_out - output_amp_val).detach()

                        # crop outputs to the region we care about
                        # output_amp_val = utils.crop_image(output_amp_val, val_res, stacked_complex=False)
                        # val_amp = utils.crop_image(val_amp, val_res, stacked_complex=False)
                        
                        # GPU上のテンソルをCPUにコピー
                        target_amp_cpu = val_amp.to('cpu')
                        loss_main_cpu = output_amp_val.to('cpu')

                        # 画像として見做した時のPSNRを計算
                        psnr_value_val = psnr(target_amp_cpu, loss_main_cpu)

                        print("val_PSNR",psnr_value_val,"@",distancebox[val_k])
                        writer.add_scalar('Val PSNR', psnr_value_val, ik)
                        output_amp0 = output_amp_val[0, ...]
                        maxVal = torch.max(output_amp0)
                        minVal = torch.min(output_amp0)
                        tmp = (output_amp0 - minVal) / (maxVal - minVal)
                        writer.add_image('Reconstruction Val Amplitude',tmp ,ik)
                        phase_generator.train()

                        # save trained model
                        if ik % 5000==0:
                            # if not os.path.isdir('checkpoints'):
                            #     os.mkdir('checkpoints')
                            torch.save(phase_generator.state_dict(),f'/images/checkall/{run_id}.pth')

                    break




                       