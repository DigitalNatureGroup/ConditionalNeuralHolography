"""

"""
import math
import cmath
import time
import numpy as np
import torch
import torch.nn as nn

import utils.utils as utils
from propagation_ASM import propagation_ASM
from utils.pytorch_prototyping.pytorch_prototyping import Conv2dSame, Unet,vunet_augumented_single,vunet_augumented


class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase

class InitialDoubleUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d):
        super(InitialDoubleUnet, self).__init__()

        net = [Unet(2, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, num_down=8, num_features_init=32, max_features=256,
                 norm=nn.BatchNorm2d, num_in=4):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet(num_in, 1, num_features_init, num_down, max_features,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase


class PhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a target amplitude"""
    def __init__(self, num_down=10, num_features_init=16, norm=nn.BatchNorm2d):
        super(PhaseOnlyUnet, self).__init__()

        net = [Unet(1, 1, num_features_init, num_down, 1024,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, target_amp):
        out_phase = self.net(target_amp)
        return (torch.ones(1), out_phase)

class ComplexUnet(nn.Module):
    """computes the final SLM phase given a target amplitude"""
    def __init__(self, num_down=10, num_features_init=16, norm=nn.BatchNorm2d):
        super(ComplexUnet, self).__init__()

        net = [Unet(2, 1, num_features_init, num_down, 1024,
                    use_dropout=False,
                    upsampling_mode='transpose',
                    norm=norm,
                    outermost_linear=True),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, target_amp):
        out_phase = self.net(target_amp)
        return (torch.ones(1), out_phase)

class VUnet_Aug_single(nn.Module):
    def __init__(self):
        super(VUnet_Aug_single,self).__init__()

        net=[vunet_augumented_single()]
        self.net=nn.Sequential(*net)

    def forward(self,input):
        out_phase=self.net([input[0],input[1]])
        return (torch.ones(1),out_phase)
    
class VUnet_Aug(nn.Module):
    def __init__(self):
        super(VUnet_Aug,self).__init__()

        net=[vunet_augumented()]
        self.net=nn.Sequential(*net)

    def forward(self,input):
        out_phase=self.net([input[0],input[1]])
        return (torch.ones(1),out_phase)




class HoloZonePlateNet(nn.Module):
    def __init__(self, initial_phase,final_phase_only,num_down=10, num_features_init=16, norm=nn.BatchNorm2d,feature_size=6.4e-6,wavelength=520e-9,linear_conv=True,distace_box=[],target_shape=[1,1,1080,1920]):
        super(HoloZonePlateNet, self).__init__()
        self.num_down=num_down
        self.num_features_init=num_features_init
        self.norm=norm
        self.feature_size = (feature_size
                        if hasattr(feature_size, '__len__')
                        else [feature_size] * 2)
        self.wavelength=wavelength
        self.linear_conv=linear_conv
        self.initial_phase=initial_phase
        self.final_phase_only=final_phase_only
        self.proptype="ASM"
        self.precomped_H = None
        self.prop = propagation_ASM
        self.dev = torch.device('cuda')
        self.distance_box=distace_box
        self.target_shape=target_shape
        self.preH_array=[]

        for c,d in enumerate(self.distance_box):
            temp_H=self.prop(torch.empty(self.target_shape, dtype=torch.complex64), self.feature_size,self.wavelength,-d, return_H=True)
            temp_H=temp_H.to(self.dev).detach()
            temp_H.requires_grad = False
            self.preH_array.append(temp_H)
            print(f"Calculating Kernel {c+1}/{len(self.distance_box)}")

    def forward(self, target,trush,ikk):
        time_counter=time.perf_counter()
        with torch.no_grad():
            
            point_light=torch.zeros(self.target_shape,dtype=torch.float32).to(self.dev)
            point_light[0, 0, self.target_shape[2]//2-1:self.target_shape[3]//2+1, self.target_shape[2]//2-1:self.target_shape[3]//2+1] = 1.0
            zone_comp= self.prop(point_light, self.feature_size,
                              self.wavelength, self.distance_box[ikk],
                              precomped_H=self.preH_array[ikk],
                              linear_conv=self.linear_conv)
            zone_plate=torch.angle(zone_comp)
            zone_plate=(zone_plate-torch.min(zone_plate))/(torch.max(zone_plate)-torch.min(zone_plate))
            # print(zone_plate)
        # print("after_plate",time.perf_counter()-time_counter)
        time_counter=time.perf_counter()
        # target_amp=x[0]
        # zone_plate=x[1]
        # distance=-x[2]
        
        init_phase = self.initial_phase(torch.cat((target, zone_plate), 1))
        real, imag = utils.polar_to_rect(target, init_phase)
        target_complex = torch.complex(real, imag)
        target_complex_diff = target_complex
        # print("after_firstUnet",time.perf_counter()-time_counter)
        time_counter=time.perf_counter()

        # implement the basic propagation to the SLM plane
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength,-self.distance_box[ikk],
                              precomped_H=self.preH_array[ikk],
                              linear_conv=self.linear_conv)
        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = utils.rect_to_polar(slm_naive.real, slm_naive.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)
        # print("after_2ndProp",time.perf_counter()-time_counter)
        time_counter=time.perf_counter()

        ret=self.final_phase_only(slm_amp_phase)
        # print("after_2ndUnet",time.perf_counter()-time_counter)
        
        return  ret
    


class HoloZonePlateNet2ch(nn.Module):
    def __init__(self, initial_phase,final_phase_only,num_down=10, num_features_init=16, norm=nn.BatchNorm2d,feature_size=6.4e-6,wavelength=520e-9,linear_conv=True,distance_box=[],target_shape=[1,1,1080,1920]):
        super(HoloZonePlateNet2ch, self).__init__()
        self.num_down=num_down
        self.num_features_init=num_features_init
        self.norm=norm
        self.feature_size = (feature_size
                        if hasattr(feature_size, '__len__')
                        else [feature_size] * 2)
        self.wavelength=wavelength
        self.linear_conv=linear_conv
        self.initial_phase=initial_phase
        self.final_phase_only=final_phase_only
        self.proptype="ASM"
        self.precomped_H = None
        self.prop = propagation_ASM
        self.dev = torch.device('cuda')
        self.target_shape=target_shape
        self.distance_box=distance_box
        self.preH_array=[]

        for c,d in enumerate(self.distance_box):
            temp_H=self.prop(torch.empty(self.target_shape, dtype=torch.complex64), self.feature_size,self.wavelength,-d, return_H=True)
            temp_H=temp_H.to(self.dev).detach()
            temp_H.requires_grad = False
            self.preH_array.append(temp_H)
            print(f"Calculating Kernel {c+1}/{len(self.distance_box)}")
        

    def forward(self, target,trush,ikk):
        complex_amp=target.to(torch.complex128)*cmath.exp(1j*distance)
        real_part=complex_amp.real
        imag_part=complex_amp.imag
        inputs=torch.cat([real_part,imag_part],1)
        inputs=inputs.to(torch.float32)

        target_amp=target
        input_img=inputs
        distance=-self.distance_box[ikk]
    
        init_phase = self.initial_phase(input_img)
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)
        target_complex_diff = target_complex
 
        # implement the basic propagation to the SLM plane
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength, distance,
                              precomped_H=self.preH_array[ikk],
                              linear_conv=self.linear_conv)
        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = utils.rect_to_polar(slm_naive.real, slm_naive.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)

        return self.final_phase_only(slm_amp_phase)




class HoloZonePlateNet_old(nn.Module):
    def __init__(self, initial_phase,final_phase_only,num_down=10, num_features_init=16, norm=nn.BatchNorm2d,feature_size=6.4e-6,wavelength=520e-9,linear_conv=True):
        super(HoloZonePlateNet_old, self).__init__()
        self.num_down=num_down
        self.num_features_init=num_features_init
        self.norm=norm
        self.feature_size = (feature_size
                        if hasattr(feature_size, '__len__')
                        else [feature_size] * 2)
        self.wavelength=wavelength
        self.linear_conv=linear_conv
        self.initial_phase=initial_phase
        self.final_phase_only=final_phase_only
        self.proptype="ASM"
        self.precomped_H = None
        self.prop = propagation_ASM
        self.dev = torch.device('cuda')

    def forward(self, x):
        target_amp=x[0]
        zone_plate=x[1]
        distance=-x[2]
        
        init_phase = self.initial_phase(torch.cat((target_amp, zone_plate), 1))
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)
        target_complex_diff = target_complex

        # implement the basic propagation to the SLM plane
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength, distance,
                              precomped_H=None,
                              linear_conv=self.linear_conv)
        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = utils.rect_to_polar(slm_naive.real, slm_naive.imag)
        slm_amp_phase = torch.cat((amp, ang), -3)

        return amp, self.final_phase_only(slm_amp_phase)