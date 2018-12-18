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


nb_plots = 32

MUAPs = np.load('./MUAPs.npy')
A = np.squeeze(np.load('./A.npy'))
z = np.squeeze(np.load('./z.npy'))

nb_plots = min(nb_plots, MUAPs.shape[0])

if nb_plots > 1:
    index_A = np.argsort(-np.sum(np.abs(A), axis=0))
    MUAPs = MUAPs[index_A]
else:
    A = np.reshape(A, (10, 1))


nb_times = A.shape[0]

# 连成长串输出
MUAPs_times = []
for j in range(nb_plots):
    muap = []
    for i in range(nb_times):
        muap = np.concatenate((muap, MUAPs[j]), 0)
    MUAPs_times.append(muap)

MUAPs_times = np.array(MUAPs_times)

# 输出MUAPs
for i in range(nb_plots):
    plt.subplot(nb_plots, 1, i+1)
    plt.plot(MUAPs_times[i])
plt.show()

'''
# 求MUAPs的间距，并用gamma函数进行拟合
MUAPs_intervals_norm = np.zeros((MUAPs.shape[0], 2))
for i in range(MUAPs.shape[0]):
    spi = spike_sorting_intervals(MUAPs[i], threshold=0.3, overlap=10, plot=False)
    interval_list = spi._interval()
    if len(interval_list) > 1:
        mu, std = stats.norm.fit(interval_list)
        MUAPs_intervals_norm[i] = [mu, std]
    elif len(interval_list) == 1:
        MUAPs_intervals_norm[i, 0] = interval_list[0]
    #print('Interval: ', interval_list)
    #print('fit gamma: ', fit_alpha, fit_loc, fit_beta)

sort_index_mu = np.argsort(MUAPs_intervals_norm[:, 0]) # 按照mu的大小进行排序
MUAPs = MUAPs[sort_index_mu]

# 输出normal distribution拟合的结果
plt.figure()
nb_muaps = MUAPs_intervals_norm.shape[0]
x_inter = np.arange(nb_muaps)
y_inter = MUAPs_intervals_norm[:, 0][sort_index_mu]
e_inter = MUAPs_intervals_norm[:, 1][sort_index_mu] 
plt.errorbar(x_inter, y_inter, e_inter, linestyle='None', marker='^')
'''

# 输出MUAPs
fig = plt.figure()
fig.suptitle('MUAPs')
for i in range(nb_plots):
    ax = plt.subplot(nb_plots, 1, i + 1)
    plt.plot(MUAPs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])

# 输出A
plt.figure()
plt.title('A')
for i in range(A.shape[0]):
    plt.bar(x=np.arange(A.shape[1]) + 0.045 * i, height=np.abs(A[i]), width=0.04, label=str(i))
plt.legend()


# 对比original, reconstruct和 remain_EMG 
original_EMG = np.squeeze(np.load('original_EMG.npy'))
reconstruct_EMG = np.squeeze(np.load('reconstruct_EMG.npy'))
nb_EMG = original_EMG.shape[0]

plt.figure()
for i in range(nb_EMG):
    plt.subplot(nb_EMG, 1, i + 1)
    plt.plot(original_EMG[i], label='EMG')
    plt.plot(reconstruct_EMG[i], '--', label='Reconstructed EMG')
    plt.ylim(-1, 1)
    #plt.legend(fontsize=15)
    plt.xticks([])
    plt.yticks([])

# 计算剩余的EMG
'''
remain_EMG = original_EMG - reconstruct_EMG
plt.figure()
plt.plot(remain_EMG + 0.5, '-.', c='c' ,label='Remaining EMG')
plt.ylim(0, 1)
'''

# 输出混合矩阵A
plt.figure()
sns.set()
#A = A[sorted_index_A]
A = np.reshape(A, (-1, 1))
#ax = sns.heatmap(A, -0.5, 0.5)
ax = sns.heatmap(A)
plt.show()