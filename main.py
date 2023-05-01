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
p.add_argument('--root_path', type=str, default='./phases', help='Directory where optimized phases will be saved.')
p.add_argument('--data_path', type=str, default='/images/div_and_flickr', help='Directory for the dataset')
p.add_argument('--val_path', type=str, default='/images/DIV2K_valid_HR', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='./pretrained_networks',
               help='Directory for the pretrained holonet/unet network')

# parse arguments
opt = p.parse_args()
TRAIN= opt.train==0
MODEL_TYPE=opt.model_type==0
DISTANCE_TO_IMAGE= opt.distance_to_image==0 
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
slm_res = (1024, 2048)  # resolution of SLM
image_res=roi_res=slm_res
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
) if DISTANCE_TO_IMAGE else HoloZonePlateNet2ch(
        wavelength=wavelength,
        feature_size=feature_size[0],
        initial_phase=InitialDoubleUnet(6, 16),
        final_phase_only=FinalPhaseOnlyUnet(8, 32, num_in=2)
)

Augmented_Conditional_Unet=VUnet_Aug_single() if DISTANCE_TO_IMAGE else VUnet_Aug()

phase_generator = Augmented_Holonet if MODEL_TYPE else Augmented_Conditional_Unet
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
val_loader =  ImageLoader(opt.data_path, channel=channel,
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
start = 0.2+wavelength/2
end = 0.2+wavelength/2+200000*wavelength
num_splits = 100

step = (end - start) / num_splits
distancebox = [start + step * i for i in range(num_splits + 1)]

for i in range(num_epochs):    
    for k, target in enumerate(image_loader):

        # get target image
        target_amp, target_res,target_filename = target
        distance=distancebox[random.randrange(num_splits)]
        ik+=1
        target_amp = target_amp.to(device)
        # distance=distance[0]
        optimizer.zero_grad()

        ## target_ampにconcat
        if DISTANCE_TO_IMAGE:
            with torch.no_grad():
                point_light=torch.zeros(1,1,image_res[0],image_res[1],dtype=torch.float32).to(device)
                point_light[0, 0, image_res[0]//2-1:image_res[0]//2+1, image_res[1]//2-1:image_res[1]//2+1] = 1.0
                model_prop = ModelPropagate(distance=distance, feature_size=feature_size, wavelength=wavelength,
                            target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                            use_conv1d_mlp=False, num_latent_codes=[0],
                            norm=None, blur=None, content_field=False, proptype="ASM").to(device)
                model_prop.eval()
                zone_plate=torch.angle(model_prop(point_light))
                zone_plate=(zone_plate-torch.min(zone_plate))/(torch.max(zone_plate)-torch.min(zone_plate))
                inputs=zone_plate
        else:
            with torch.no_grad():
                complex_amp=target_amp.to(torch.complex128)*cmath.exp(1j*distance)
                real_part=complex_amp.real
                imag_part=complex_amp.imag
                inputs=torch.cat([real_part,imag_part],1)
                inputs=inputs.to(torch.float32)

        # forward model
        slm_amp, slm_phase = phase_generator([target_amp,inputs,distance])


        model_prop = ModelPropagate(distance=distance, feature_size=feature_size, wavelength=wavelength,
                            target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                            use_conv1d_mlp=False, num_latent_codes=[0],
                            norm=None, blur=None, content_field=False, proptype="ASM").to(device)


        model_prop.eval()  # ensure freezing propagation model
        output_complex = model_prop(slm_phase)

        output_lin_intensity = torch.sum(output_complex.abs()**2 * 0.95, dim=1, keepdim=True)
        output_amp = torch.pow(output_lin_intensity, 0.5)

        target_res=target_res[0]

        # crop outputs to the region we care about
        # output_amp = utils.crop_image(output_amp, target_res, stacked_complex=False)         
        # target_amp = utils.crop_image(target_amp, target_res, stacked_complex=False)
        

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

        print(f'iteration {ik}:Loss:{loss_main.item()} PSNR:{psnr_value}w/{target_filename}@{distance}')

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
                        val_distance=distancebox[random.randrange(num_splits)]

                        if DISTANCE_TO_IMAGE:
                            ## target_ampにconcat
                            val_point_light=torch.zeros(1,1,image_res[0],image_res[1],dtype=torch.float32).to(device)
                            val_point_light[0, 0, image_res[0]//2-1:image_res[0]//2+1, image_res[1]//2-1:image_res[1]//2+1] = 1.0
                            model_prop_val = ModelPropagate(distance=val_distance, feature_size=feature_size, wavelength=wavelength,
                                        target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                                        use_conv1d_mlp=False, num_latent_codes=[0],
                                        norm=None, blur=None, content_field=False, proptype="ASM").to(device)
                            model_prop_val.eval()
                            val_zone_plate=torch.angle(model_prop_val(val_point_light))
                            val_zone_plate=(val_zone_plate-torch.min(val_zone_plate))/(torch.max(val_zone_plate)-torch.min(val_zone_plate))
                        else:
                            complex_amp=target_amp.to(torch.complex128)*cmath.exp(1j*distance)
                            real_part=complex_amp.real
                            imag_part=complex_amp.imag
                            inputs=torch.cat([real_part,imag_part],1)
                            inputs=inputs.to(torch.float32)
                            val_zone_plate=inputs

                        # forward model

                        _,val_phase=phase_generator([val_amp,val_zone_plate,val_distance])


                    
                        ### model_propが画像によって変わっていく

                        prop_model_val = ModelPropagate(distance=val_distance, feature_size=feature_size, wavelength=wavelength,
                                            target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                                            use_conv1d_mlp=False, num_latent_codes=[0],
                                            norm=None, blur=None, content_field=False, proptype="ASM").to(device)

                        prop_model_val.eval()  # ensure freezing propagation model
                        output_complex_val = prop_model_val(val_phase)


                        output_lin_intensity_val = torch.sum(output_complex_val.abs()**2 * 0.95, dim=1, keepdim=True)

                        output_amp_val = torch.pow(output_lin_intensity_val, 0.5)


                        # crop outputs to the region we care about
                        # output_amp_val = utils.crop_image(output_amp_val, val_res, stacked_complex=False)
                        # val_amp = utils.crop_image(val_amp, val_res, stacked_complex=False)
                        
                        # GPU上のテンソルをCPUにコピー
                        target_amp_cpu = val_amp.to('cpu')
                        loss_main_cpu = output_amp_val.to('cpu')

                        # 画像として見做した時のPSNRを計算
                        psnr_value_val = psnr(target_amp_cpu, loss_main_cpu)

                        print("val_PSNR",psnr_value_val,"@",val_distance)
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




                       