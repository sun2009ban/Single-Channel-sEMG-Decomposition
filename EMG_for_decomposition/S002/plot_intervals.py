'''
输出MUAPTs的时间间隔的图片
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('E:\swt\Spike_Sorting')
from Spike_Sorting import spike_sorting

import scipy.stats as stats

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


MUAPs = np.load('./S00201_MUAP_resample.npy')

plt.figure()
plt.plot(MUAPs[1])
plt.show()


intervals = []

# 求MUAPs的间距
for i in range(MUAPs.shape[0]):
    spi = spike_sorting_intervals(MUAPs[i], threshold=0.1, overlap=10, plot=False)
    interval_list = spi._interval()
    if len(interval_list) > 1:
        intervals.extend(interval_list)    

intervals = np.array(intervals)

fit_alpha, fit_loc, fit_scale = stats.gamma.fit(intervals)
print('alpha: {}, loc: {}, scale: {}'.format(fit_alpha, fit_loc, fit_scale))

n, bins, patches = plt.hist(intervals, 10, density=True)
# 显示gamma分布曲线
x = range(0,351,2)
y = stats.gamma.pdf(x, fit_alpha, fit_loc, fit_scale)
plt.plot(x, y, 'r--', linewidth=3)

plt.ylim(0, 0.025)
plt.xticks(range(0, 351, 50))
plt.show()