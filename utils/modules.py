"""
Some modules for easy use. (No need to calculate kernels explicitly)

"""
import torch
import torch.nn as nn
from algorithms import gerchberg_saxton, stochastic_gradient_descent, double_phase_amplitude_coding
from datetime import datetime

import os
import time
import skimage.io
import utils.utils as utils
import platform
my_os = platform.system()
# if True:
#     from utils.arduino_laser_control_module import ArduinoLaserControl
#     from utils.camera_capture_module import CameraCapture
#     from utils.calibration_module import Calibration
#     from utils.slm_display_module import SLMDisplay


class GS(nn.Module):
    """Classical Gerchberg-Saxton algorithm

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> gs = GS(...)
    >>> final_phase = gs(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self,distance_box, target_shape,wavelength, feature_size, phase_path=None,
                 prop_model='ASM', propagator=None, writer=None, device=torch.device('cuda')):
        super(GS, self).__init__()

        # Setting parameters
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.phase_path = phase_path
        self.prop_model = prop_model
        self.prop = propagator
        self.writer = writer
        self.dev = device
        self.distancebox=distance_box
        self.target_shape=target_shape


    def forward(self, target_amp, init_phase=None,ikk=0,num_iters=1,preH=None,preHb=None):

        # Run algorithm
        final_phase = gerchberg_saxton(init_phase, target_amp, num_iters, self.distancebox[ikk],
                                       self.wavelength, self.feature_size,
                                       phase_path=self.phase_path,
                                       prop_model=self.prop_model, propagator=self.prop,
                                       precomputed_H_f=preH, precomputed_H_b=preHb,
                                       writer=self.writer)
        return final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path


class SGD(nn.Module):
    """Proposed Stochastic Gradient Descent Algorithm using Auto-diff Function of PyTorch

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param num_iters: the number of iteration, default 500
    :param roi_res: region of interest to penalize the loss
    :param phase_path: path to write intermediate results
    :param loss: loss function, default L2
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param lr: learning rate for phase variables
    :param lr_s: learning rate for the learnable scale
    :param s0: initial scale
    :param writer: SummaryWrite instance for tensorboard
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> sgd = SGD(...)
    >>> final_phase = sgd(target_amp, init_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    init_phase: initial guess of phase of phase-only slm
    final_phase: optimized phase-only representation at SLM plane, same dimensions
    """
    def __init__(self, distancebox,kernelbox,wavelength,feature_size, roi_res, phase_path=None, prop_model='ASM',
                 propagator=None, loss=nn.MSELoss(), lr=0.01, lr_s=0.003, s0=1.0, citl=False, camera_prop=None,
                 writer=None, device=torch.device('cuda')):
        super(SGD, self).__init__()

        # Setting parameters
        self.distancebox=distancebox
        self.kernelbox=kernelbox
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.roi_res = roi_res
        self.phase_path = phase_path
        self.prop_model = prop_model
        self.prop = propagator

        self.lr = lr
        self.lr_s = lr_s
        self.init_scale = s0

        self.citl = citl
        self.camera_prop = camera_prop
        self.preHarray=[]
        self.writer = writer
        self.dev = device
        print("loss",loss)
        self.loss = loss.to(device)
       



    def forward(self, target_amp,init_phase=None,ikk=0,num_iters=1,preH=None):
        # Pre-compute propagataion kernel only once
        # print("req",preH.requires_grad)

        file_path = '/code/for_paper/record.txt'

        # 追加する文字列
        string_to_add = f"{datetime.now()}\n{preH}\n"

        # ファイルに文字列を追加する関数
        def append_to_file(file_path, string_to_add):
            # 'a'モードでファイルを開き、文字列を追加します
            with open(file_path, 'a') as file:
                file.write(string_to_add)

        # 関数を呼び出してファイルに文字列を追加
        append_to_file(file_path, string_to_add)

        # Run algorithm
        final_phase = stochastic_gradient_descent(init_phase, target_amp,num_iters, self.distancebox[ikk],
                                                  self.wavelength, self.feature_size,
                                                  roi_res=self.roi_res, phase_path=self.phase_path,
                                                  prop_model=self.prop_model, propagator=self.prop,
                                                  loss=self.loss, lr=self.lr, lr_s=self.lr_s, s0=self.init_scale,
                                                  citl=self.citl, camera_prop=self.camera_prop,
                                                  writer=self.writer,
                                                  precomputed_H=preH)
        
        print(torch.cuda.memory_allocated())
        print(torch.cuda.max_memory_allocated())
        
        torch.cuda.empty_cache()
    
        return final_phase

    @property
    def init_scale(self):
        return self._init_scale

    @init_scale.setter
    def init_scale(self, s):
        self._init_scale = s

    @property
    def citl_hardwares(self):
        return self._citl_hardwares

    @citl_hardwares.setter
    def citl_hardwares(self, citl_hardwares):
        self._citl_hardwares = citl_hardwares

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        self._prop = prop


class DPAC(nn.Module):
    """Double-phase Amplitude Coding

    Class initialization parameters
    -------------------------------
    :param prop_dist: propagation dist between SLM and target, in meters
    :param wavelength: the wavelength of interest, in meters
    :param feature_size: the SLM pixel pitch, in meters
    :param prop_model: chooses the propagation operator ('ASM': propagation_ASM,
        'model': calibrated model). Default 'ASM'.
    :param propagator: propagator instance (function / pytorch module)
    :param device: torch.device

    Usage
    -----
    Functions as a pytorch module:

    >>> dpac = DPAC(...)
    >>> _, final_phase = dpac(target_amp, target_phase)

    target_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]
    target_amp (optional): phase at the target plane, with dimensions [batch, 1, height, width]
    final_phase: optimized phase-only representation at SLM plane, same dimensions

    """
    def __init__(self,distancebox,kernelbox,wavelength, feature_size, prop_model='ASM', propagator=None,
                 device=torch.device('cuda')):
        """

        """
        super(DPAC, self).__init__()

        # propagation is from target to SLM plane (one step)
        self.distancebox=distancebox
        self.kernelbox=kernelbox
        self.wavelength = wavelength
        self.feature_size = feature_size
        self.prop_model = prop_model
        self.prop = propagator
        self.dev = device

     

    def forward(self, target_amp, target_phase=None,ikk=0,preH=None):
        if target_phase is None:
            target_phase = torch.zeros_like(target_amp)

        file_path = '/code/for_paper/record.txt'

        # 追加する文字列
        string_to_add = f"{datetime.now()}\{-self.distancebox[ikk]}\n{preH}\n"

        # ファイルに文字列を追加する関数
        def append_to_file(file_path, string_to_add):
            # 'a'モードでファイルを開き、文字列を追加します
            with open(file_path, 'a') as file:
                file.write(string_to_add)

        # 関数を呼び出してファイルに文字列を追加
        append_to_file(file_path, string_to_add)    


        final_phase = double_phase_amplitude_coding(target_phase, target_amp, -self.distancebox[ikk],
                                                    self.wavelength, self.feature_size,
                                                    prop_model=self.prop_model, propagator=self.prop,
                                                    # precomputed_H=self.kernelbox[ikk].to(self.dev),
                                                    precomputed_H=preH)
        return final_phase

    @property
    def phase_path(self):
        return self._phase_path

    @phase_path.setter
    def phase_path(self, phase_path):
        self._phase_path = phase_path



class PhysicalProp(nn.Module):
    """ A module for physical propagation,
    forward pass displays gets SLM pattern as an input and display the pattern on the physical setup,
    and capture the diffraction image at the target plane,
    and then return warped image using pre-calibrated homography from instantiation.

    Class initialization parameters
    -------------------------------
    :param channel:
    :param slm_settle_time:
    :param roi_res: *** Note that the order of x / y is reversed here ***
    :param num_circles:
    :param laser_arduino:
    :param com_port:
    :param arduino_port_num:
    :param range_row:
    :param range_col:
    :param patterns_path:
    :param calibration_preview:

    Usage
    -----
    Functions as a pytorch module:

    >>> camera_prop = PhysicalProp(...)
    >>> captured_amp = camera_prop(slm_phase)

    slm_phase: phase at the SLM plane, with dimensions [batch, 1, height, width]
    captured_amp: amplitude at the target plane, with dimensions [batch, 1, height, width]

    """
    def __init__(self, channel=1, slm_settle_time=0.1, roi_res=(1600, 880), num_circles=(21, 12),
                 laser_arduino=False, com_port='COM3', arduino_port_num=(6, 10, 11),
                 range_row=(200, 1000), range_col=(300, 1700),
                 patterns_path=f'F:/citl/calibration', show_preview=False):
        super(PhysicalProp, self).__init__()

        # 1. Connect Camera
        self.camera = CameraCapture()
        self.camera.connect(0)  # specify the camera to use, 0 for main cam, 1 for the second cam

        # 2. Connect SLM
        self.slm = SLMDisplay()
        self.slm.connect()
        self.slm_settle_time = slm_settle_time

        # 3. Connect to the Arduino that switches rgb color through the laser control box.
        if laser_arduino:
            self.alc = ArduinoLaserControl(com_port, arduino_port_num)
            self.alc.switch_control_box(channel)
        else:
            self.alc = None

        # 4. Calibrate hardwares using homography
        calib_ptrn_path = os.path.join(patterns_path, f'{("red", "green", "blue")[channel]}.png')
        space_btw_circs = [int(roi / (num_circs - 1)) for roi, num_circs in zip(roi_res, num_circles)]

        self.calibrate(calib_ptrn_path, num_circles, space_btw_circs,
                       range_row=range_row, range_col=range_col, show_preview=show_preview)

    def calibrate(self, calibration_pattern_path, num_circles, space_btw_circs,
                  range_row, range_col, show_preview=False, num_grab_images=10):
        """
        pre-calculate the homography between target plane and the camera captured plane

        :param calibration_pattern_path:
        :param num_circles:
        :param space_btw_circs: number of pixels between circles
        :param slm_settle_time:
        :param range_row:
        :param range_col:
        :param show_preview:
        :param num_grab_images:
        :return:
        """

        self.calibrator = Calibration(num_circles, space_btw_circs)

        # supposed to be a grid pattern image (21 x 12) for calibration
        calib_phase_img = skimage.io.imread(calibration_pattern_path)
        self.slm.show_data_from_array(calib_phase_img)

        # sleep for 0.1s
        time.sleep(self.slm_settle_time)

        # capture displayed grid pattern image
        captured_intensities = self.camera.grab_images(num_grab_images)  # capture 5-10 images for averaging
        captured_img = utils.burst_img_processor(captured_intensities)

        # masking out dot pattern region for homography
        captured_img_masked = captured_img[range_row[0]:range_row[1], range_col[0]:range_col[1], ...]
        calib_success = self.calibrator.calibrate(captured_img_masked, show_preview=show_preview)

        self.calibrator.start_row, self.calibrator.end_row = range_row
        self.calibrator.start_col, self.calibrator.end_col = range_col

        if calib_success:
            print('   - calibration success')
        else:
            raise ValueError('  - Calibration failed')

    def forward(self, slm_phase, num_grab_images=1):
        """
        this forward pass gets slm_phase to display and returns the amplitude image at the target plane.

        :param slm_phase:
        :param num_grab_images:
        :return: A pytorch tensor shape of (1, 1, H, W)
        """

        slm_phase_8bit = utils.phasemap_8bit(slm_phase, True)

        # display the pattern and capture linear intensity, after perspective transform
        captured_linear_np = self.capture_linear_intensity(slm_phase_8bit, num_grab_images=num_grab_images)

        # convert raw-16 linear intensity image into an amplitude tensor
        if len(captured_linear_np.shape) > 2:
            captured_linear = torch.tensor(captured_linear_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            captured_linear = captured_linear.to(slm_phase.device)
            captured_linear = torch.sum(captured_linear, dim=1, keepdim=True)
        else:
            captured_linear = torch.tensor(captured_linear_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            captured_linear = captured_linear.to(slm_phase.device)

        # return amplitude
        return torch.sqrt(captured_linear)

    def capture_linear_intensity(self, slm_phase, num_grab_images):
        """

        :param slm_phase:
        :param num_grab_images:
        :return:
        """

        # display on SLM and sleep for 0.1s
        self.slm.show_data_from_array(slm_phase)
        time.sleep(self.slm_settle_time)

        # capture and take average
        grabbed_images = self.camera.grab_images(num_grab_images)
        captured_intensity_raw_avg = utils.burst_img_processor(grabbed_images)  # averaging

        # crop ROI as calibrated
        captured_intensity_raw_cropped = captured_intensity_raw_avg[
            self.calibrator.start_row:self.calibrator.end_row,
            self.calibrator.start_col:self.calibrator.end_col, ...]
        # apply homography
        return self.calibrator(captured_intensity_raw_cropped)

    def disconnect(self):
        self.camera.disconnect()
        self.slm.disconnect()
        if self.alc is not None:
            self.alc.turnOffAll()
