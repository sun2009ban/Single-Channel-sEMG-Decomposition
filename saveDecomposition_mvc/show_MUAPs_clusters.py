import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('E:\swt\Spike_Sorting')
from Spike_Sorting import spike_sorting

import scipy.stats as stats

'''
分类好的MUAPs_clusters进行spike time的组合
'''


class spike_sorting_intervals(spike_sorting):
    def __init__(self, raw_data, fs=1000, cutoff=100, threshold=0.1, overlap=10, save_interval_path=None, 
                    before_max=5, after_max=5, plot=False):
        super(spike_sorting_intervals, self).__init__(raw_data, fs, cutoff, threshold, overlap, save_interval_path, before_max, after_max, plot)
    def _get_dict(self):
        # 数据高通滤波
        filtered_data = self.raw_data
        # 提取spike，把性质存储在dict中
        spikes_dict = self._spike_threshold(filtered_data, self.threshold, self.overlap, self.plot)
        # 计算和存储intervals
        return spikes_dict

MUAPs = np.load('./MUAPs_centers.npy').item() # 这个是个dict

nb_clusters = len(MUAPs.keys())

for i in range(nb_clusters):
    for muaps in MUAPs[str(i)]:
        spi = spike_sorting_intervals(muaps, threshold=0.5, overlap=10, plot=False)
        spi_dict = spi._get_dict()
        spike_times = []
        ax = plt.subplot(nb_clusters, 1, i+1)
        if len(spi_dict) > 0:
            for spike in spi_dict.keys():
                spike_times.append(spi_dict[spike][2])
            plt.plot(muaps)
            plt.vlines(spike_times, 0, 1)    
            plt.xlim(0, muaps.shape[0])
            plt.ylim(-1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks([])
            plt.yticks([])

plt.show()