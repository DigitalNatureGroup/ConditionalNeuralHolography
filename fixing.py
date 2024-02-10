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
    kernel_a=KernelLoader(f"/images/kernels/0.2_200000_100_1024_3.74e-06_back")

    kernel_c=propagation_ASM(u_in,(3.74*1e-6,3.74*1e-6),520*1e-9,-0.20000104000000002,return_H=True,precomped_H=None)
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
    

def check_sampling_theorem(dx, dy, wavelength, z):
    # 許容される最大サンプリング間隔を計算
    max_sampling_interval_x = wavelength * z / (dx * 2)
    max_sampling_interval_y = wavelength * z / (dy * 2)

    # サンプリング間隔のチェック
    if dx > max_sampling_interval_x or dy > max_sampling_interval_y:
        return "サンプリング間隔が大きすぎます。標本化定理に違反している可能性があります。"

    # 伝搬距離のチェック（任意の追加条件があればここに記述）
    # ...

    return "サンプリング間隔と伝搬距離は標本化定理を満たしています。"

# 使用例
# dx = 3.74*1e-6  # 例: 0.1 mm
# dy = 3.74*1e-6  # 例: 0.1 mm
# wavelength = 520*1e-9  # 例: 500 nm
# z = 0.2 # 例: 10 cm

check_kernel()

    