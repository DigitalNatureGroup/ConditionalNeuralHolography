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
import csv
import time
import os
import random
import cmath
import sys
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

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument('--channel', type=int, default=1, help='Red:0, green:1, blue:2')
p.add_argument('--gen_type', type=int, default=0, help='CNN:0,DPAC:1,SGD:2,GS:3')
p.add_argument('--model_type', type=int, default=0, help='Augmented Holonet:0, Augmented Conditional Unet:1')
p.add_argument('--distance_to_image', type=int, default=0, help='Zone Plate:0, Reflect Changed Phase:1')
p.add_argument('--start_dis', type=float, default=0.2, help='z_0[m]')
p.add_argument('--alpha', type=float, default=2.0, help='phase_shift')
p.add_argument('--end_dis', type=int, default=200000, help='end of distances')
p.add_argument('--num_split', type=int, default=100, help='number of distance points')
# p.add_argument('--num_iters', type=int, default=1, help='number of iteraion used in GS & SGD')
p.add_argument('--root_path', type=str, default='/images/compare', help='Directory where optimized phases will be saved.')
p.add_argument('--kernel_path', type=str, default='/images/kernels', help='Directory where optimized phases will be saved.')
p.add_argument('--plate_path', type=str, default='/images/zoneplates', help='Directory where optimized phases will be saved.')
p.add_argument('--val_path', type=str, default='/images/DIV2K_valid_HR', help='Directory for the dataset')
p.add_argument('--generator_dir', type=str, default='/images/checkall',
               help='Directory for the pretrained holonet/unet network')

opt = p.parse_args()
gen_type=opt.gen_type


# parse arguments

MODEL_TYPE=opt.model_type==0
DISTANCE_TO_IMAGE= opt.distance_to_image==0 
status_name="Eval"

num_splits=opt.num_split

model_type_name="Augmented_Holonet" if MODEL_TYPE else "Augmented_Conditional_Unet"
distance_to_image_name="Zone_Plate" if DISTANCE_TO_IMAGE else "Reflect_Changed_Phase"
gen_string=["CNN","DPAC","SGD","GS","Old"][gen_type]
start_dis=opt.start_dis
alpha=opt.alpha
num_splits=opt.num_split
end_dis=opt.end_dis
run_id = f"{status_name}_{model_type_name}_{distance_to_image_name}_{start_dis}_{end_dis}_{num_splits}" if (gen_type==0 or gen_type==4) else gen_string
channel = opt.channel  # Red:0 / Green:1 / Blue:2
chan_str = ('red', 'green', 'blue')[channel]
print("getntype",gen_type)
print(f'   - eavaluating phase with {run_id} ... ')

# Hyperparameters setting
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
prop_dist = (20 * cm, 20 * cm, 20 * cm)[channel]  # propagation distance from SLM plane to target plane
wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]  # wavelength of each color
feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
slm_res = (1024, 2048)  # resolution of SLM
image_res=slm_res
roi_res=(880,1600)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device('cuda')  # The gpu you are using
print("device",device)
print("count",torch.cuda.device_count())

# Options for the algorithm


s0 = 0.95  # initial scaled

root_path = os.path.join(opt.root_path, run_id, chan_str)  # path for saving out optimized phases
kernel_path=os.path.join(opt.kernel_path,f'{start_dis}_{end_dis}_{num_splits}')
plate_path=os.path.join(opt.plate_path,f'{start_dis}_{end_dis}_{num_splits}')

# Tensorboard writer
summaries_dir = os.path.join(root_path, 'summaries')
utils.cond_mkdir(summaries_dir)
writer = SummaryWriter(summaries_dir)
propagator = propagation_ASM  # Ideal model
###################

def delete_file(path):
    """Given a path to a file, delete the file if it exists"""
    if os.path.isfile(path):
        os.remove(path)
        print(f"File {path} has been deleted.")
    else:
        print(f"No such file: {path}")

# ファイルパスを指定
dd_path = f'{summaries_dir}/{start_dis}_{end_dis}_{num_splits}.csv'
delete_file(dd_path)



# loads images from disk, set to augment with flipping
val_loader =  ImageLoader(opt.val_path, channel=channel,
                        image_res=image_res, homography_res=roi_res,
                        crop_to_homography=True,
                        shuffle=False, vertical_flips=False, horizontal_flips=False)
print(image_res,roi_res)

ik=0
start = start_dis+wavelength/alpha
end = start_dis+wavelength/alpha+end_dis*wavelength
# num_splits = 10

step = (end - start) / num_splits
distancebox = [start + step * i for i in range(num_splits + 1)]
## カーネルが既にあるかどうかを確認
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

init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)


if gen_type==0:
    
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
        distance_box=distancebox,
        ) if DISTANCE_TO_IMAGE else VUnet_Aug(target_shpae=[1,1,slm_res[0],slm_res[1]],
        feature_size=feature_size,
        wavelength=wavelength,
        distance_box=distancebox)

    phase_only_algorithm = Augmented_Holonet if MODEL_TYPE else Augmented_Conditional_Unet
    weight_path=f"{opt.generator_dir}/{run_id.replace('Eval','Train')}.pth"
    phase_only_algorithm.load_state_dict(torch.load(weight_path))
    phase_only_algorithm.to(device)
    phase_only_algorithm.eval()  

elif gen_type==4:

    phase_generator=HoloZonePlateNet_old(
        wavelength=wavelength,
        feature_size=feature_size[0],
        initial_phase=InitialDoubleUnet(6, 16),
        final_phase_only=FinalPhaseOnlyUnet(8, 32, num_in=2),
    )   

    weight_path=f"{opt.generator_dir}/{run_id.replace('Eval','Train')}.pth"
    phase_generator.load_state_dict(torch.load(weight_path))
    phase_generator.to(device)
    phase_generator.eval()  

elif gen_type==1:
    phase_only_algorithm = DPAC(distancebox,kLoader,wavelength, feature_size, "ASM", propagator, device)
elif gen_type==2:
    
    phase_only_algorithm = SGD(distancebox,kLoader,wavelength, feature_size,roi_res, root_path,
                    'ASM',propagation_ASM, nn.MSELoss().to(device), 8e-3, 2e-3, s0, False, None, writer, device)
elif gen_type==3:
    phase_only_algorithm = GS(distancebox, [0,0,slm_res[0],slm_res[1]],wavelength, feature_size,root_path,"ASM", propagation_ASM, writer, device)


if(gen_type==2 or gen_type==3):
    num_iters_array=[5]
else :
    num_iters_array=[1]

log_interval = 500  # Adjust this value to your needs
log_records = []



for l,num_iters in enumerate(num_iters_array):
    ik=0

    for target in (val_loader):


        for k in range(num_splits):
            if k%(num_splits//100)==0:

                # get target image
                target_amp, target_res,target_filename = target
                distance=distancebox[k]
                ik+=1
                target_amp = target_amp.to(device)
                # distance=distance[0]
                # slm_phase=0


                computing_time=0 



                if gen_type==4:
            
                    with torch.no_grad():
                        start_time=time.perf_counter()
                        ## target_ampにconcat
                        if DISTANCE_TO_IMAGE:
                            with torch.no_grad():
                                point_light=torch.zeros(1,1,image_res[0],image_res[1],dtype=torch.float32).to(device)
                                print("zp_dis",distance)
                                point_light[0, 0, image_res[0]//2-1:image_res[0]//2+1, image_res[1]//2-1:image_res[1]//2+1] = 1.0
                                model_prop = ModelPropagate(distance=distance, feature_size=feature_size, wavelength=wavelength,
                                            target_field=False, num_gaussians=0, num_coeffs_fourier=0,
                                            use_conv1d_mlp=False, num_latent_codes=[0],
                                            norm=None, blur=None, content_field=False, proptype="ASM").to(device)
                                model_prop.eval()
                                zone_plate=torch.angle(model_prop(point_light))
                                zone_plate=(zone_plate-torch.min(zone_plate))/(torch.max(zone_plate)-torch.min(zone_plate))
                                inputs=zone_plate
                                print(zone_plate)
                        else:
                            
                                complex_amp=target_amp.to(torch.complex128)*cmath.exp(1j*distance)
                                real_part=complex_amp.real
                                imag_part=complex_amp.imag
                                inputs=torch.cat([real_part,imag_part],1)
                                inputs=inputs.to(torch.float32)

                        # forward model
                        slm_amp, slm_phase = phase_generator([target_amp,inputs,distance])
                        computing_time=time.perf_counter()-start_time
                #use DPAC
                elif gen_type== 1:
                    pass
                    # phase_only_algorithm = DPAC(distance, wavelength, feature_size, "ASM", propagator, device)
                    # start_time=time.perf_counter()
                    # _,slm_phase=phase_only_algorithm(target_amp)
                    # computing_time=time.perf_counter()-start_time
                
                #use SGD
                elif gen_type == 2:
                    # init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
                    # phase_only_algorithm = SGD(distance, wavelength, feature_size, num_iters, roi_res, root_path,
                    #                 'ASM',propagation_ASM, nn.MSELoss().to(device), 8e-3, 2e-3, s0, False, None, writer, device)
                    pass
                    # start_time=time.perf_counter()
                    # slm_phase=phase_only_algorithm(target_amp,init_phase)
                    # computing_time=time.perf_counter()-start_time
                #use GS
                elif gen_type == 3:
                    pass
                    # init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *slm_res)).to(device)
                    # phase_only_algorithm = GS(distance, wavelength, feature_size, num_iters, root_path,
                    #                 "ASM", propagation_ASM, writer, device)
                    # start_time=time.perf_counter()
                    # slm_phase=phase_only_algorithm(target_amp,init_phase)
                    # computing_time=time.perf_counter()-start_time


                #use ASM
                
                # phase_only_algorithm.init_scale = s0 * utils.crop_image(target_amp, roi_res, stacked_complex=False).mean()
                # phase_only_algorithm.phase_path = os.path.join(root_path)

                preH=kLoader[k].to(device)
                preH.requires_grad=False
                preHb=kbLoder[k].to(device)
                preHb.requires_grad=False
                plate=plateLoader[k].to(device)
                plate.requires_grad=False

                start_time=time.perf_counter() 
                # print(phase_only_algorithm)
            
                if gen_type==0:
                    with torch.no_grad():
                        slm_phase = phase_only_algorithm(target_amp,plate,k,preHb)
                elif gen_type==1:
                    with torch.no_grad():
                        slm_phase = phase_only_algorithm(target_amp,init_phase,k,preHb)
                elif gen_type==2:
                    slm_phase = phase_only_algorithm(target_amp,init_phase,k,num_iters,preH)
                elif gen_type==3:
                    with torch.no_grad():
                        slm_phase = phase_only_algorithm(target_amp,init_phase,k,num_iters,preH,preHb)
                elif gen_type==4:
                    pass

                with torch.no_grad():
                    runtime=time.perf_counter()-start_time
                    print("runtime",runtime)
                    record_time=runtime


                    real, imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
                    slm_field = torch.complex(real, imag)

                    output_complex=utils.propagate_field(slm_field,propagator,distance,wavelength,feature_size,"ASM",dtype = torch.float32,precomputed_H=preH)

                    # output_lin_intensity = torch.sum(output_complex.abs()**2 * 0.95, dim=1, keepdim=True)
                    # output_amp = torch.pow(output_lin_intensity, 0.5)
                    target_amp = utils.crop_image(target_amp, target_shape=roi_res, stacked_complex=False).to(device)

                    output_amp=output_complex.abs()
                    recon_amp=utils.crop_image(output_amp, target_shape=roi_res, stacked_complex=False)
                    
                    ## これ解明
                    recon_amp *= (torch.sum(recon_amp * target_amp, (-2, -1), keepdim=True)
                        / torch.sum(recon_amp * recon_amp, (-2, -1), keepdim=True))


                    target_res=target_res[0]

                    ## PSNRを計算

                    def psnr(tensor1, tensor2):
                        mse = F.mse_loss(tensor1, tensor2)
                        max_pixel_value = 1
                        psnr = 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())
                        return psnr
                    
                    # GPU上のテンソルをCPUにコピー
                    target_amp_cpu = target_amp[0,0,:,:].to('cpu')
                    loss_main_cpu =recon_amp[0,0,:,:].to('cpu')

                    # 画像として見做した時のPSNRを計算
                    psnr_value = psnr(target_amp_cpu, loss_main_cpu)

                    print(f'image_No {ik} PSNR:{psnr_value}w/{target_filename}@{distance}')
                    log_records.append([ik, psnr_value])


                    if ik % log_interval == 0:
                        with open(f'{summaries_dir}/{start_dis}_{end_dis}_{num_splits}.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(log_records)
                            log_records = []
                            print(f'Saved log at iteration: {ik}')


    # # writer.close()