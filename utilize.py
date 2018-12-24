import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('/home/swt/Documents/PythonProject/Spike_Sorting')
from Spike_Sorting import spike_sorting

import scipy.stats as stats

'''
里面存储各种的辅助操作函数
unity function
'''

class spike_sorting_intervals(spike_sorting):
    def __init__(self, raw_data, fs=1000, cutoff=100, threshold=0.1, overlap=10, save_interval_path=None, 
					before_max=5, after_max=5, plot=False):
        super(spike_sorting_intervals, self).__init__(raw_data, fs, cutoff, threshold, overlap, save_interval_path, before_max, after_max, plot)

    def _interval(self):
        # 数据高通滤波
	    filtered_data = self.raw_data
	    # 提取spike，把性质存储在dict中
	    spikes_dict = self._spike_threshold(filtered_data, self.threshold, self.overlap, self.plot)
	    # 计算和存储intervals
	    return self._get_intervals(spikes_dict, self.save_interval_path, self.plot)

    def _get_dict(self):
        # 数据高通滤波
        filtered_data = self.raw_data
        # 提取spike，把性质存储在dict中
        spikes_dict = self._spike_threshold(filtered_data, self.threshold, self.overlap, self.plot)
        # 计算和存储intervals
        return spikes_dict


def cal_intervals_and_fit_with_gamma(MUAPs, threshold=0.2, overlap=10, plot=False, cutoff=350):
    '''
    计算MUAPs的spike时间间隔，同时利用gamma拟合时间间隔的分布

    MUAPs: n x data_dims， n 是MUAPs的数目 
    threshold: 提取spike的阈值
    overlap: 两个spike间的距离
    cutoff: 两个spike间最长的interval的值
    
    intervals: 一个numpy array，里面是所有MUAPs的spike intervals的值
    fig: 是绘制有intervals 的 hist 图，以及 gamma 分布的拟合
    '''
    intervals = []

    # 求MUAPs的间距
    for i in range(MUAPs.shape[0]):
        spi = spike_sorting_intervals(MUAPs[i], threshold=threshold, overlap=overlap, plot=False)
        interval_list = spi._interval()
        if len(interval_list) > 1:
            intervals.extend(interval_list)    

    intervals = np.array(intervals)
    intervals = intervals[intervals < cutoff]

    fig = None
    if plot:
        fig = plt.figure()
        fit_alpha, fit_loc, fit_scale = stats.gamma.fit(intervals)
        print('alpha: {}, loc: {}, scale: {}'.format(fit_alpha, fit_loc, fit_scale))

        n, bins, patches = plt.hist(intervals, 10, density=True)

        # 显示gamma分布曲线
        x = range(0,351,2)
        y = stats.gamma.pdf(x, fit_alpha, fit_loc, fit_scale)
        plt.plot(x, y, 'r--', linewidth=3)

        plt.ylim(0, 0.025)
        plt.xticks(range(0, 351, 50))
        
    return intervals, fig


def plot_spike_time(nb_plots, MUAPs, threshold=0.02, overlap=10, plot_raw=False, linewidth=2):
    '''
    绘制MUAPs的spike time图，每个spike time用vertical line标注出来
    nb_plots: 最多绘制多少个MUAPs出来
    '''

    nb_plots = min(nb_plots, MUAPs.shape[0])

    fig = plt.figure()

    for i in range(nb_plots):
        spi = spike_sorting_intervals(MUAPs[i], threshold=threshold, overlap=overlap, plot=False)
        spi_dict = spi._get_dict()
        spike_times = []
        ax = plt.subplot(nb_plots, 1, i+1)
        if len(spi_dict) > 0:
            for spike in spi_dict.keys():
                spike_times.append(spi_dict[spike][2])
            if plot_raw:
                plt.plot(MUAPs[i])   
            plt.xlim(0, MUAPs.shape[1])
            plt.ylim(-1, 1)
            plt.vlines(spike_times, 0, 1, linewidth=linewidth) 
            plt.hlines(0, 0, MUAPs.shape[1], linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks([])
            plt.yticks([])

    return fig


def plot_MUAPs(nb_plots, MUAPs, A=None):
    
    nb_plots = min(nb_plots, MUAPs.shape[0])

    fig = plt.figure()

    for i in range(nb_plots):
        ax = plt.subplot(nb_plots, 1, i + 1)

        if A is None:
            plt.plot(MUAPs[i])
        else:
            plt.plot(A[i] * MUAPs[i])
        
        plt.ylim(-1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)        
        plt.xticks([])
        plt.yticks([])

    return fig



def find_top_n_MUAPs_according_to_A(A, MUAPs, threshold, n):
	'''
	A，按照A的值进行排序,A要求是一个vector 长度是>= n
	MUAPs, 是一个 n x data_dims 的矩阵，data_dims是时间长度
	n是选出top多少出来
	'''
	index = np.argsort(-A) # 从大到小排列

	MUAPs_new = []
	A_new = []

	count = 0
	for i in index:
		if np.any(np.abs(A[i] * MUAPs[i]) > threshold):
			MUAPs_new.append(MUAPs[i])
			A_new.append(A[i])
			count += 1

		if count >= n:
			break

	return np.array(MUAPs_new), np.array(A_new)


def plot_EMG_and_its_approximation(original_EMG, approximated_EMG):
    # 对比original, approximated和 remain_EMG 
    assert original_EMG.shape == approximated_EMG.shape, "Shape of the EMG and its approximation should be the same."

    nb_EMG = original_EMG.shape[0]

    plt.figure()
    for i in range(nb_EMG):
        plt.subplot(nb_EMG, 1, i + 1)
        plt.plot(original_EMG[i], label='EMG')
        plt.plot(approximated_EMG[i], '--', label='Reconstructed EMG')
        plt.ylim(-1, 1)
        #plt.legend(fontsize=15)
        plt.xticks([])
        plt.yticks([])

    # 计算剩余的EMG

    remain_EMG = original_EMG - approximated_EMG
    
    plt.figure()
    for i in range(nb_EMG):
        plt.subplot(nb_EMG, 1, i + 1)
        plt.plot(remain_EMG[i], '-.', c='c' ,label='Remaining EMG')
        plt.ylim(-1, 1)

    return remain_EMG