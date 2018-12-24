# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# IIR FIlter
from scipy.signal import butter, lfilter, freqz, iirnotch

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch(cutoff, fs, Q=None):
    w0 = cutoff / (fs / 2)
    if Q is None:
        Q = 3/w0 # 质量系数
    assert w0 > 0 and w0 <= 1
    b, a = iirnotch(w0, Q)
    w, h = freqz(b, a)
    # Generate frequency axis
    freq = w*fs/(2*np.pi)
    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(freq, 20*np.log10(abs(h)), color='blue')
    ax.set_title("Frequency Response")
    ax.set_ylabel("Amplitude (dB)", color='blue')
    ax.set_ylim([-25, 10])
    ax.grid()
    return

def notch_filter(data, cutoff, fs):
    '''
    陷波滤波器，cutoff为被限制频率，值为0 < cutoff < (fs/2)
    '''
    w0 = cutoff / (fs / 2)
    Q = min(3/w0, 3) # 质量系数
    assert w0 > 0 and w0 <= 1
    b, a = iirnotch(w0, Q)
    y = lfilter(b, a, data)
    return y


# FIR Filter，设计为矩阵的变换的形式
# 设计FIR滤波器
from scipy import signal

def fir_filter_matrix(data_dims, cutoff=100, fs=1000, order=17):
    '''
    data_dims是滤波器作用的数据的长度
    返回值是一个matrix，filtered_data = np.matmul(data, coeff_matrix)
    ''' 
    fir_coeff = signal.firwin(order, cutoff, pass_zero=False, fs=fs)
    fir_coeff = fir_coeff[::-1]

    coeff_matrix = np.zeros((data_dims, data_dims)) #和数据长度对应
    fir_coeff_vector = np.zeros(data_dims)

    for i in range(order):
        fir_coeff_vector[i] = fir_coeff[i]

    for j in range(order, data_dims):
        coeff_matrix[:, j] = np.roll(fir_coeff_vector, j - order)

    return coeff_matrix