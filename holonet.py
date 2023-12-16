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
    def __init__(self,target_shpae,feature_size,wavelength,distance_box,linear_conv=True):
        super(VUnet_Aug_single,self).__init__()

        self.target_shape=target_shpae
        self.feature_size=feature_size
        self.wavelength=wavelength
        self.distance_box=distance_box
        self.linear_conv=linear_conv
        self.dev=torch.device("cuda")
        self.prop = propagation_ASM
        net=[vunet_augumented_single()]
        self.net=nn.Sequential(*net)

    def forward(self,input,trush,ikk,preHb):

        with torch.no_grad():
            point_light=torch.zeros(self.target_shape,dtype=torch.float32).to(self.dev)
            point_light[0, 0, self.target_shape[2]//2-1:self.target_shape[2]//2+1, self.target_shape[3]//2-1:self.target_shape[3]//2+1] = 1.0
            zone_comp= self.prop(point_light, self.feature_size,
                                self.wavelength, self.distance_box[ikk],
                                precomped_H=preHb,
                                linear_conv=self.linear_conv)
            zone_plate=torch.angle(zone_comp)
            zone_plate=(zone_plate-torch.min(zone_plate))/(torch.max(zone_plate)-torch.min(zone_plate))

        out_phase=self.net([input,zone_plate])
        return out_phase
    
class VUnet_Aug(nn.Module):
    def __init__(self,target_shpae,feature_size,wavelength,distance_box,linear_conv=True):
        super(VUnet_Aug,self).__init__()

        self.target_shape=target_shpae
        self.feature_size=feature_size
        self.wavelength=wavelength
        self.distance_box=distance_box
        self.linear_conv=linear_conv
        self.dev=torch.device("cuda")
        self.prop = propagation_ASM
        net=[vunet_augumented_single()]
        self.net=nn.Sequential(*net)

        net=[vunet_augumented()]
        self.net=nn.Sequential(*net)

    def forward(self,target,trush,ikk,preHb):
        distance=-self.distance_box[ikk]
        complex_amp=target.to(torch.complex128)*cmath.exp(1j*distance)
        real_part=complex_amp.real
        imag_part=complex_amp.imag
        inputs=torch.cat([real_part,imag_part],1)
        inputs=inputs.to(torch.float32)

        target_amp=target
        input_img=inputs


        out_phase=self.net([target,inputs])
        return out_phase




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

        # for c,d in enumerate(self.distance_box):
        #     temp_H=self.prop(torch.empty(self.target_shape, dtype=torch.complex64), self.feature_size,self.wavelength,-d, return_H=True)
        #     temp_H=temp_H.to(self.dev).detach()
        #     temp_H.requires_grad = False
        #     self.preH_array.append(temp_H)
        #     print(f"Calculating Kernel {c+1}/{len(self.distance_box)}")

    def forward(self, target,trush,ikk,preHb):
        time_counter=time.perf_counter()
       
            # print(zone_plate)
        # print("after_plate",time.perf_counter()-time_counter)
        time_counter=time.perf_counter()
        # target_amp=x[0]
        # zone_plate=x[1]
        # distance=-x[2]
        
        init_phase = self.initial_phase(torch.cat((target, trush), 1))
        real, imag = utils.polar_to_rect(target, init_phase)
        target_complex = torch.complex(real, imag)
        target_complex_diff = target_complex
        # print("after_firstUnet",time.perf_counter()-time_counter)
        time_counter=time.perf_counter()

        # implement the basic propagation to the SLM plane
       
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength,-self.distance_box[ikk],
                              precomped_H=preHb,
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


    def forward(self, target,trush,ikk,preH):
        distance=-self.distance_box[ikk]
        complex_amp=target.to(torch.complex128)*cmath.exp(1j*distance)
        real_part=complex_amp.real
        imag_part=complex_amp.imag
        inputs=torch.cat([real_part,imag_part],1)
        inputs=inputs.to(torch.float32)

        target_amp=target
        input_img=inputs
        
    
        init_phase = self.initial_phase(input_img)
        real, imag = utils.polar_to_rect(target_amp, init_phase)
        target_complex = torch.complex(real, imag)
        target_complex_diff = target_complex
 
        # implement the basic propagation to the SLM plane
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength, distance,
                              precomped_H=preH,
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
    
class HoloNet(nn.Module):
    """Generates phase for the final non-iterative model

    Class initialization parameters
    -------------------------------
    distance: propagation dist between SLM and target, in meters, default 0.1.
        Note: distance is negated internally, so the PhaseGenerator and
        ProcessAndPropagate get the same input
    wavelength: the wavelength of interest, in meters, default 520e-9
    feature_size: the SLM pixel pitch, in meters, default 6.4e-6
    zernike_coeffs: a torch tensor that corresponds to process_phase.py,
        ProcessAndPropagate.coeffs, after training is completed. Default None,
        which disables passing zernike coeffs to the final network
    source_amplitude: a process_phase.SourceAmplitude module, after training.
        Default None, which disables passing source amp to the final network
    target_field: a torch tensor that corresponds to propagation_model.py,
        citl_calibrated_model.target_field, after training is completed. Default None,
        which disables passing target_field to the final network
    latent_codes: a citl_calibrated_model.latent_codes parameter, after training.
        Default None, which disables passing latent_codes to the final network
    initial_phase: a module that returns an initial phase given the target amp.
        Default None, which assumes all zeros initial phase
    final_phase_only: a module that processes the post-propagation amp+phase to
        a phase-only output that works as well as iterative results. Default
        None, which switches to double phase coding
    proptype: chooses the propagation operator ('ASM': propagation_ASM,
        'fresnel': propagation_fresnel). Default ASM.
    linear_conv: if True, pads for linear conv for propagation. Default True

    Usage
    -----
    Functions as a pytorch module:

    >>> phase_generator = HoloNet(...)
    >>> slm_amp, slm_phase = phase_generator(target_amp)

    target_amp: amplitude at the target plane, with dimensions [batch, 1,
        height, width]
    slm_amp: amplitude to be encoded in the phase pattern at the SLM plane. Used
        to enforce uniformity, if desired. Same as target dimensions
    slm_phase: encoded phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, distance=0.1, wavelength=520e-9, feature_size=6.4e-6,
                 zernike_coeffs=None, source_amplitude=None, target_field=None, latent_codes=None,
                 initial_phase=None, final_phase_only=None, proptype='ASM', linear_conv=True,
                 manual_aberr_corr=False):
        super(HoloNet, self).__init__()

        # submodules
        self.source_amplitude = source_amplitude
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only
        if target_field is not None:
            self.target_field = target_field.detach()
        else:
            self.target_field = None

        if latent_codes is not None:
            self.latent_codes = latent_codes.detach()
        else:
            self.latent_codes = None

        # propagation parameters
        self.wavelength = wavelength
        self.feature_size = (feature_size
                             if hasattr(feature_size, '__len__')
                             else [feature_size] * 2)
        self.distance = -distance

        self.zernike_coeffs = (None if zernike_coeffs is None
                               else -zernike_coeffs.clone().detach())

        # objects to precompute
        self.zernike = None
        self.precomped_H = None
        self.precomped_H_zernike = None
        self.source_amp = None

        # whether to pass zernike/source amp as layers or divide out manually
        self.manual_aberr_corr = manual_aberr_corr

        # make sure parameters from the model training phase don't update
        if self.zernike_coeffs is not None:
            self.zernike_coeffs.requires_grad = False
        if self.source_amplitude is not None:
            for p in self.source_amplitude.parameters():
                p.requires_grad = False

        # change out the propagation operator
        if proptype == 'ASM':
            self.prop = propagation_ASM
        else:
            ValueError(f'Unsupported prop type {proptype}')

        self.linear_conv = linear_conv

        # set a device for initializing the precomputed objects
 
        self.dev = torch.device("cuda")
    
    def forward(self, target_amp):
        # compute some initial phase, convert to real+imag representation
        if self.initial_phase is not None:
            init_phase = self.initial_phase(target_amp)
            real, imag = utils.polar_to_rect(target_amp, init_phase)
            target_complex = torch.complex(real, imag)
        else:
            init_phase = torch.zeros_like(target_amp)
            # no need to convert, zero phase implies amplitude = real part
            target_complex = torch.complex(target_amp, init_phase)

        # subtract the additional target field
        if self.target_field is not None:
            target_complex_diff = target_complex - self.target_field
        else:
            target_complex_diff = target_complex

        # precompute the propagation kernel only once
        if self.precomped_H is None:
            self.precomped_H = self.prop(target_complex_diff,
                                         self.feature_size,
                                         self.wavelength,
                                         self.distance,
                                         return_H=True,
                                         linear_conv=self.linear_conv)
            self.precomped_H = self.precomped_H.to(self.dev).detach()
            self.precomped_H.requires_grad = False

        if self.precomped_H_zernike is None:
            if self.zernike is None and self.zernike_coeffs is not None:
                self.zernike_basis = compute_zernike_basis(self.zernike_coeffs.size()[0],
                                                           [i * 2 for i in target_amp.size()[-2:]], wo_piston=True)
                self.zernike_basis = self.zernike_basis.to(self.dev).detach()
                self.zernike = combine_zernike_basis(self.zernike_coeffs, self.zernike_basis)
                self.zernike = utils.ifftshift(self.zernike)
                self.zernike = self.zernike.to(self.dev).detach()
                self.zernike.requires_grad = False
                self.precomped_H_zernike = self.zernike * self.precomped_H
                self.precomped_H_zernike = self.precomped_H_zernike.to(self.dev).detach()
                self.precomped_H_zernike.requires_grad = False
            else:
                self.precomped_H_zernike = self.precomped_H

        # precompute the source amplitude, only once
        if self.source_amp is None and self.source_amplitude is not None:
            self.source_amp = self.source_amplitude(target_amp)
            self.source_amp = self.source_amp.to(self.dev).detach()
            self.source_amp.requires_grad = False

        # implement the basic propagation to the SLM plane
        slm_naive = self.prop(target_complex_diff, self.feature_size,
                              self.wavelength, self.distance,
                              precomped_H=self.precomped_H_zernike,
                              linear_conv=self.linear_conv)

        # switch to amplitude+phase and apply source amplitude adjustment
        amp, ang = utils.rect_to_polar(slm_naive.real, slm_naive.imag)
        # amp, ang = slm_naive.abs(), slm_naive.angle()  # PyTorch 1.7.0 Complex tensor doesn't support
                                                         # the gradient of angle() currently.

        if self.source_amp is not None and self.manual_aberr_corr:
            amp = amp / self.source_amp

        if self.final_phase_only is None:
            return amp, double_phase(amp, ang, three_pi=False)
        else:
            # note the change to usual complex number stacking!
            # We're making this the channel dim via cat instead of stack
            if (self.zernike is None and self.source_amp is None
                    or self.manual_aberr_corr):
                if self.latent_codes is not None:
                    slm_amp_phase = torch.cat((amp, ang, self.latent_codes.repeat(amp.shape[0], 1, 1, 1)), -3)
                else:
                    slm_amp_phase = torch.cat((amp, ang), -3)
            elif self.zernike is None:
                slm_amp_phase = torch.cat((amp, ang, self.source_amp), -3)
            elif self.source_amp is None:
                slm_amp_phase = torch.cat((amp, ang, self.zernike), -3)
            else:
                slm_amp_phase = torch.cat((amp, ang, self.zernike,
                                           self.source_amp), -3)
            return amp, self.final_phase_only(slm_amp_phase)

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.zernike is not None:
            slf.zernike = slf.zernike.to(*args, **kwargs)
        if slf.precomped_H is not None:
            slf.precomped_H = slf.precomped_H.to(*args, **kwargs)
        if slf.source_amp is not None:
            slf.source_amp = slf.source_amp.to(*args, **kwargs)
        if slf.target_field is not None:
            slf.target_field = slf.target_field.to(*args, **kwargs)
        if slf.latent_codes is not None:
            slf.latent_codes = slf.latent_codes.to(*args, **kwargs)

        # try setting dev based on some parameter, default to cpu
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf