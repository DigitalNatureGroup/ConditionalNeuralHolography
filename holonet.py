"""

"""
import math
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
    def __init__(self, initial_phase,final_phase_only,num_down=10, num_features_init=16, norm=nn.BatchNorm2d,feature_size=6.4e-6,wavelength=520e-9,linear_conv=True):
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
    


class HoloZonePlateNet2ch(nn.Module):
    def __init__(self, initial_phase,final_phase_only,num_down=10, num_features_init=16, norm=nn.BatchNorm2d,feature_size=6.4e-6,wavelength=520e-9,linear_conv=True):
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

    def forward(self, x):
        target_amp=x[0]
        input_img=x[1]
        distance=-x[2]
    
        init_phase = self.initial_phase(input_img)
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
