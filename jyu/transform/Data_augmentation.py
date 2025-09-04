# -*- coding: UTF-8 -*-
import numpy as np
import random
from scipy.interpolate import interp1d


class TRandomSelector:
    """
    随机的选择一个transform分支，类名中T表示transform
    """

    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def __call__(self, x):
        trans = random.choice(self.transforms)
        return trans(x)

class TSelector:
    """
    根据概率选择一个transform分支，类名中T表示transform
    """

    def __init__(self, transforms: list, probabilities: list):
        super().__init__()
        self.transforms = transforms
        self.p = probabilities

    def __call__(self, x):
        trans = random.choices(self.transforms, weights=self.p, k=1)[0]
        # trans = np.random.choice(self.transforms, p=self.p)  # 要求概率和为1
        return trans(x)

class ReturnData:
    def __call__(self, x, *args, **kwds):
        '''直接将数据返回'''
        return x

class GaussianNoise:
    """
    高斯噪声
    注入随机噪声，模拟传感器误差。
    """
    def __init__(self, mean=0, std=0.05) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = np.random.normal(self.mean, self.std, x.shape)
        x = x + noise
        return x

class SPNoise:
    """添加椒盐噪声"""
    def __call__(self, x):
        output = np.zeros(x.shape ,x.dtype)
        prob = random.uniform(0.0005,0.001)  #随机噪声比例
        thres = 1 - prob
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = x[i][j]
        return output

class Reverse:
    """反转"""
    def __call__(self, x):
        return self.reverse(x)

    @staticmethod
    def reverse(x: np):
        x = np.flip(x).copy() # copy() 确保内存连续
        return x

class ScaleAmplitude:
    """
    时间缩放(Scaling) - 幅度缩放：调整信号的整体或局部幅值。
    """
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, signal):
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return signal * scale_factor

class TimeShift:
    """
    时间平移(Shifting) - 整体平移：沿时间轴平移信号，边界填充（如镜像或常数）。
    """
    def __init__(self, shift_max=10):
        self.shift_max = shift_max

    def __call__(self, signal):
        shift_max = self.shift_max
        shift = np.random.randint(-shift_max, shift_max)
        shifted_signal = np.roll(signal, shift)
        # 处理边界（填充0或边缘值）
        if shift > 0:
            shifted_signal[:shift] = signal[-1]  # 填充末尾值
        elif shift < 0:
            shifted_signal[shift:] = signal[0]   # 填充起始值
        return shifted_signal

class WindowWarp:
    """
    窗口扭曲(Window Warping)：对随机时间窗口进行局部缩放或拉伸。
    """
    def __init__(self, window_ratio=0.1, scale_range=(0.5, 2.0)):
        self.window_ratio = window_ratio
        self.scale_range = scale_range
    
    def __call__(self, x):
        return self.window_warp(x, self.window_ratio, self.scale_range)

    @staticmethod
    def window_warp(signal, window_ratio=0.1, scale_range=(0.5, 2.0)):
        if len(signal) == 1:
            signal = signal.reshape(1, -1)
        length = signal.shape[1]
        window_size = int(length * window_ratio)
        start = np.random.randint(0, max(1, length - window_size))  # 避免window_size为0时出错
        end = start + window_size

        # 生成缩放后的时间轴
        original_time = np.arange(length)
        warped_time = original_time.copy()
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_duration = window_size * scale
        new_end = start + scaled_duration
        warped_time[start:end] = np.linspace(start, new_end, window_size)

        # 插值重构信号
        interp_fn = interp1d(original_time, signal, kind='linear')
        warped_signal = interp_fn(np.clip(warped_time, 0, length-1))
        return warped_signal


# ================================================================ #
class SliceSplice:
    """
    时间切片与拼接（Slicing & Splicing）
    从同一信号的不同部分或不同信号中截取片段拼接。
    """
    def __init__(self, num_slices=3):
        self.num_slices = num_slices

    def __call__(self, x):
        splits = np.array_split(x, self.num_slices)
        np.random.shuffle(splits)
        return np.concatenate(splits)

class FrequencyPerturb:
    """
    频域增强
    在频域添加扰动（如滤波、相位偏移）。
    """
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, x):
        fft = np.fft.fft(x)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        # 在幅度上添加噪声
        magnitude *= np.random.normal(1, self.noise_level, len(magnitude))
        modified_fft = magnitude * np.exp(1j * phase)
        return np.fft.ifft(modified_fft).real
