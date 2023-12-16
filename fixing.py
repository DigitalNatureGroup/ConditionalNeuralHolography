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
from utils.augmented_image_loader import ImageLoader
from utils.kernel_loader import KernelLoader
from propagation_model import ModelPropagate
from holonet import *
from propagation_ASM import propagation_ASM

res=[1024,2048]
u_in=torch.zeros((1,1,res[0],res[1]),dtype=torch.float32)


def check_kernel():
    ## Kernelの精度を確認する
    kernel_a=KernelLoader(f"/images/kernels/0.2_200000_100_1024_3.74e-06")

    kernel_c=propagation_ASM(u_in,(3.74*1e-6,3.74*1e-6),520*1e-9,0.20000104000000002,return_H=True,precomped_H=None)
    torch.set_printoptions(threshold=500)

    print("kernelA \n",kernel_a[0])
    print("    ")

    print("kernelC \n",kernel_c)

    print(type(kernel_a[0]))

    print(type(kernel_c))


    print("等しいか",torch.equal(kernel_a[0],kernel_c))


def check_plate():
    plate_a=KernelLoader("/images/zoneplates/0.2_200000_100_1024_3.74e-06")[0]
    # point_light=torch.zeros([1,1,slm_res[0],slm_res[1]],dtype=torch.float32).to(device)
    # point_light[0, 0, slm_res[0]//2-1:slm_res[0]//2+1, slm_res[1]//2-1:slm_res[1]//2+1] = 1.0
    u_in[0,0,res[0]//2-1:res[0]//2+1,res[1]//2-1:res[1]//2+1]=1.0
    plate_c=propagation_ASM(u_in,(3.74*1e-6,3.74*1e-6),520*1e-9,-0.20000104000000002)
    plate_c=torch.angle(plate_c)
    plate_c=(plate_c-torch.min(plate_c))/(torch.max(plate_c)-torch.min(plate_c))   
    plate_c.to("cuda")   
    

    print(plate_a)
  
    print(plate_c)
    print("等しいか",torch.equal(plate_a,plate_c))
    
check_plate()
    