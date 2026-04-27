import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES
import h5py
import io

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadH5Data(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 magnitude: bool = False,
                 magnitude_mean_std: bool = False,
                 magnitude_dB: bool = False,
                 magnitude_dB_norm: bool = False,
                 magnitude_dB_mean_std: bool = False,
                 magnitude_dB_no_norm_mean_std: bool = False,
                 power: bool = False,
                 dB_t75_power: bool = False,
                 IQ_abs: bool = False,
                 IQ_abs_3_norm: bool = False,
                 IQ_abs_log_norm: bool = False,
                 only_I: bool = False,
                 IQ_abs_3_symbol: bool = False,
                 IQ_abs_mean_std: bool = False,
                 i_q_mean_std: bool = False,
                 IQ_chai_4: bool = False,
                 IQ_chai_4_biaozhunhua: bool = False,
                 IQ_convert_0_1: bool = False,
                 IQ_abs_log: bool = False,
                 complex: bool = False,
                 phase_dB_abs: bool = False,
                 phase_dB: bool = False,
                 phase_dB_concate: bool = False,
                 dB_i_q: bool = False,
                 to_float32: bool = False,
                 dataset_key: str = 'dataset',
                 ignore_empty: bool = False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):

        self.ignore_empty = ignore_empty
        self.magnitude = magnitude
        self.magnitude_mean_std = magnitude_mean_std
        self.magnitude_dB = magnitude_dB
        self.magnitude_dB_norm = magnitude_dB_norm
        self.magnitude_dB_mean_std = magnitude_dB_mean_std
        self.magnitude_dB_no_norm_mean_std = magnitude_dB_no_norm_mean_std
        self.power = power
        self.dB_t75_power = dB_t75_power
        self.IQ_abs = IQ_abs
        self.IQ_abs_log_norm = IQ_abs_log_norm
        self.only_I = only_I
        self.IQ_abs_3_norm = IQ_abs_3_norm
        self.IQ_abs_3_symbol = IQ_abs_3_symbol
        self.IQ_chai_4 = IQ_chai_4
        self.IQ_chai_4_biaozhunhua = IQ_chai_4_biaozhunhua
        self.IQ_abs_mean_std = IQ_abs_mean_std
        self.i_q_mean_std =  i_q_mean_std
        self.IQ_convert_0_1 = IQ_convert_0_1
        self.IQ_abs_log = IQ_abs_log
        self.to_float32 = to_float32
        self.phase_dB_abs = phase_dB_abs
        self.phase_dB = phase_dB
        self.phase_dB_concate = phase_dB_concate
        self.dB_i_q = dB_i_q
        self.dataset_key = dataset_key
        self.complex = complex

        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None



        if self.complex and (self.IQ_abs or self.IQ_abs_log):
            raise ValueError("'complex' cannot be True with 'IQ_abs' or 'IQ_abs_log'")
        if self.IQ_abs and self.IQ_abs_log:
            raise ValueError("'IQ_abs' and 'IQ_abs_log' cannot be True at the same time")


    def normalize_and_combine_channels(self, I_channel, Q_channel, num_stages=3, decay_factor=0.1):


        all_normalized = []


        current_max = np.max(I_channel)
        current_min = np.min(I_channel)
        denominator = current_max - current_min


        if denominator < 1e-12:
            denominator = 1e-12

        for _ in range(num_stages):
            normalized = (I_channel - current_min) / denominator
            all_normalized.append(np.clip(normalized, 0, 1))
            current_max *= decay_factor
            denominator = current_max - current_min  #

        #
        current_max = np.max(Q_channel)
        current_min = np.min(Q_channel)
        denominator = current_max - current_min

        if denominator < 1e-12:
            denominator = 1e-12

        for _ in range(num_stages):
            normalized = (Q_channel - current_min) / denominator
            all_normalized.append(np.clip(normalized, 0, 1))
            current_max *= decay_factor
            denominator = current_max - current_min  # 更新分母

        #
        return np.stack(all_normalized, axis=-1)


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        try:
            #
            if self.file_client_args is not None:

                file_bytes = self.file_client.get(filename)  #
                h5_file = h5py.File(io.BytesIO(file_bytes), 'r')  #
            else:
                h5_file = h5py.File(filename, 'r')  #

            with h5_file:  #
                complex_data = h5_file[self.dataset_key][()]

            #
            if complex_data.ndim != 2 or complex_data.shape != (512, 512):
                raise ValueError(f"Expected complex array shape (512, 512), but got {complex_data.shape}")


            if self.complex:
                img = np.array(complex_data, dtype=np.complex64)
            else:
                if self.magnitude:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)

                elif self.magnitude_mean_std:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    mean = 0.0689536
                    std = 0.09862229
                    norm_magnitude = (magnitude - mean) / std

                elif self.power:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    power = real_part ** 2 + imag_part ** 2

                elif self.magnitude_dB_norm:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    power_dB = 20 * np.log10(magnitude)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min)

                elif self.magnitude_dB:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    power_dB = 20 * np.log10(magnitude)


                elif self.magnitude_dB_mean_std:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    epsilon = 1e-10
                    power_dB = 20 * np.log10(magnitude)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min + epsilon)

                    uint8 = norm_power_dB * 255
                    mean = 168.1604999359131
                    std = 24.787222272822767
                    norm_power_dB = (uint8 - mean) / std


                elif self.magnitude_dB_no_norm_mean_std:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = real_part ** 2 + imag_part ** 2
                    power_dB = 10 * np.log10(magnitude)
                    mean = -30.084263589092878
                    std = 11.404058539922564
                    norm_power_dB = (power_dB - mean) / std

                elif self.dB_t75_power:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
                    power = magnitude ** 2

                    power_dB = 20 * np.log10(magnitude)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min) * 255

                    t75 = np.percentile(norm_power_dB, 75)
                    dB_75 = np.where(norm_power_dB > t75, norm_power_dB, 0)
                    power_norm = (power - np.min(power)) / (np.max(power) - np.min(power)) * 255


                elif self.IQ_abs:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)

                elif self.IQ_abs_3_norm:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)

                elif self.IQ_abs_3_symbol:
                    real_part = np.abs(complex_data.real)
                    imag_part = np.abs(complex_data.imag)
                    sign_I = np.sign(complex_data.real)
                    sign_Q = np.sign(complex_data.imag)
                    # 符号组合编码：1: (+, +), 2: (+, -), 3: (-, +), 4: (-, -)
                    sign_combo = np.zeros_like(real_part, dtype=real_part.dtype)
                    sign_combo[(sign_I == 1) & (sign_Q == 1)] = 1
                    sign_combo[(sign_I == 1) & (sign_Q == -1)] = 2
                    sign_combo[(sign_I == -1) & (sign_Q == 1)] = 3
                    sign_combo[(sign_I == -1) & (sign_Q == -1)] = 4

                elif self.IQ_abs_mean_std:
                    real_part = (np.abs(complex_data.real) - 0.043895974475695564) / 0.07288876349643252
                    imag_part = (np.abs(complex_data.imag) - 0.04389960850433757) / 0.07289885064199604

                elif self.IQ_convert_0_1:
                    real_part = (complex_data.real + 1) / 2
                    imag_part = (complex_data.imag + 1) / 2

                elif self.IQ_abs_log:
                    real_part = 20 * np.log(np.abs(complex_data.real))
                    imag_part = 20 * np.log(np.abs(complex_data.imag))

                elif self.IQ_abs_log_norm:
                    real_dB = 20 * np.log(np.abs(complex_data.real))
                    imag_dB = 20 * np.log(np.abs(complex_data.imag))

                    real_dB_min = np.min(real_dB)
                    real_dB_max = np.max(real_dB)
                    real_part = (real_dB - real_dB_min) / (real_dB_max - real_dB_min)

                    imag_dB_min = np.min(imag_dB)
                    imag_dB_max = np.max(imag_dB)
                    imag_part = (imag_dB - imag_dB_min) / (imag_dB_max - imag_dB_min)

                elif self.IQ_chai_4:
                    real_part = complex_data.real
                    imag_part = complex_data.imag

                    I_mask_z = np.where(real_part > 0, real_part, 0)
                    I_mask_f = np.where(real_part < 0, real_part, 0)
                    I_mask_f = np.abs(I_mask_f)

                    Q_mask_z = np.where(imag_part > 0, imag_part, 0)
                    Q_mask_f = np.where(imag_part < 0, imag_part, 0)
                    Q_mask_f = np.abs(Q_mask_f)

                elif self.IQ_chai_4_biaozhunhua:
                    real_part = complex_data.real
                    imag_part = complex_data.imag

                    I_mask_z = np.where(real_part > 0, real_part, 0)
                    I_mask_f = np.where(real_part < 0, real_part, 0)
                    I_mask_f = np.abs(I_mask_f)

                    Q_mask_z = np.where(imag_part > 0, imag_part, 0)
                    Q_mask_f = np.where(imag_part < 0, imag_part, 0)
                    Q_mask_f = np.abs(Q_mask_f)

                    I_mask_z = (I_mask_z - 0.021947987236305318) / 0.056018242196901304
                    I_mask_f = (I_mask_f - 0.021947987236305318) / 0.056018242196901304
                    Q_mask_z = (Q_mask_z - 0.021947987236305318) / 0.056018242196901304
                    Q_mask_f = (Q_mask_f - 0.021947987236305318) / 0.056018242196901304


                elif self.phase_dB:

                    magnitude = np.abs(complex_data)
                    phase_i = np.real(complex_data / magnitude)
                    phase_q = np.imag(complex_data / magnitude)

                    epsilon = 1e-10
                    magnitude = np.maximum(magnitude, epsilon)
                    power_dB = 20 * np.log10(magnitude ** 2 + epsilon)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min + epsilon)
                    real_part = phase_i * norm_power_dB
                    imag_part = phase_q * norm_power_dB

                elif self.phase_dB_concate:

                    i = np.real(complex_data)
                    q = np.imag(complex_data)
                    theta = np.arctan(i, q)
                    mean_theta = -2.105743294720836e-05
                    std_theta = 1.8138022296756446
                    normlize_theta = (theta - mean_theta) / std_theta

                    magnitude = np.abs(complex_data)
                    epsilon = 1e-10
                    magnitude = np.maximum(magnitude, epsilon)
                    power_dB = 20 * np.log10(magnitude ** 2 + epsilon)

                    # 标准化
                    mean_power_dB = -60.168423331432265
                    std_power_dB = 22.807646139669906
                    norm_power_dB = (power_dB - mean_power_dB) / std_power_dB

                    real_part = norm_power_dB
                    imag_part = normlize_theta

                elif self.phase_dB_abs:
                    magnitude = np.abs(complex_data)
                    phase_i = np.real(complex_data / magnitude)
                    phase_q = np.imag(complex_data / magnitude)

                    epsilon = 1e-10
                    magnitude = np.maximum(magnitude, epsilon)
                    power_dB = 20 * np.log10(magnitude ** 2 + epsilon)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min + epsilon)
                    real_part = np.abs(phase_i * norm_power_dB)
                    imag_part = np.abs(phase_q * norm_power_dB)

                elif self.dB_i_q:

                    magnitude = np.abs(complex_data)
                    phase_i = np.real(complex_data / magnitude)
                    phase_q = np.imag(complex_data / magnitude)
                    real_part = phase_i
                    imag_part = phase_q

                    epsilon = 1e-10
                    magnitude = np.maximum(magnitude, epsilon)
                    power_dB = 20 * np.log10(magnitude ** 2 + epsilon)
                    power_dB_min = np.min(power_dB)
                    power_dB_max = np.max(power_dB)
                    norm_power_dB = (power_dB - power_dB_min) / (power_dB_max - power_dB_min + epsilon)
                elif self.i_q_mean_std:
                    real_part = complex_data.real / 0.088
                    imag_part = complex_data.imag / 0.088
                else:
                    real_part = complex_data.real
                    imag_part = complex_data.imag

                if self.dB_i_q:
                    # print(np.min(norm_power_dB),np.max(norm_power_dB))
                    # print(np.min(real_part),np.max(real_part))
                    # print(np.min(imag_part),np.max(imag_part))
                    # print('=======')
                    img = np.stack([norm_power_dB, real_part, imag_part], axis=-1)

                elif self.IQ_chai_4 or self.IQ_chai_4_biaozhunhua:

                    img = np.stack([I_mask_z, I_mask_f, Q_mask_z, Q_mask_f], axis=-1)
                else:
                    img = np.stack([real_part, imag_part], axis=-1)  # 形状：(512, 512, 2)

                if self.IQ_abs_3_norm:
                    img = self.normalize_and_combine_channels(real_part, imag_part)

                if self.IQ_abs_3_symbol:
                    img = np.stack([real_part, imag_part, sign_combo], axis=-1)

                if self.only_I:
                    img = complex_data.real

                if self.magnitude:
                    img = magnitude

                if self.magnitude_dB:
                    img = power_dB

                if self.magnitude_dB_norm:
                    img = norm_power_dB

                if self.magnitude_dB_mean_std:
                    img = norm_power_dB

                if self.magnitude_dB_no_norm_mean_std:
                    img = norm_power_dB
                if self.dB_t75_power:
                    img = np.stack([norm_power_dB, dB_75, power_norm], axis=-1)

                if self.magnitude_mean_std:
                    img = norm_magnitude

                if self.power:
                    img = power


        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e


        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles(object):
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadProposals(object):
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations(object):
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results


@PIPELINES.register_module()
class LoadRPDV2Annotations(object):
    """Load mutiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self):
        super(LoadRPDV2Annotations, self).__init__()

    def _load_semantic_map_from_box(self, results):
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        pad_shape = results['pad_shape']
        gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_sem_map = np.zeros((80, int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.float32)
        gt_sem_weights = np.zeros((80, int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.float32)

        indexs = np.argsort(gt_areas)
        for ind in indexs[::-1]:
            box = gt_bboxes[ind]
            box_mask = np.zeros((int(pad_shape[0] / 8), int(pad_shape[1] / 8)), dtype=np.int64)
            box_mask[int(box[1] / 8):int(box[3] / 8) + 1, int(box[0] / 8):int(box[2] / 8) + 1] = 1
            gt_sem_map[gt_labels[ind]][box_mask > 0] = 1
            gt_sem_weights[gt_labels[ind]][box_mask > 0] = 1 / gt_areas[ind]

        results['gt_sem_map'] = gt_sem_map
        results['gt_sem_weights'] = gt_sem_weights

        return results

    def __call__(self, results):
        """Call function to load multiple types annotations

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        results = self._load_semantic_map_from_box(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox_semantic_map={True}, '
        return repr_str