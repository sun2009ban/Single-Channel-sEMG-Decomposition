import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys

import scipy.stats as stats

'''
这个主要是把这么多的MUAPs和A排列成spikes的形式
'''

nb_select = 80 # 只显示最大的几个值

MUAPs = np.load('./h_c.npy')
A = np.squeeze(np.load('./A_c.npy'))
p_MUAPs = np.load('./p_c.npy')

print('MUAPs: ', MUAPs.shape)
print('A: ', A.shape)
print('MUAPs Probability: ', p_MUAPs.shape)

pdb.set_trace()

# 每一段都按照自己最大的排列，直接粘合
'''
index_A = np.argsort(-np.abs(A), axis=1) # 加上负号是为了使其从大到小排列
index_A = index_A[:, :nb_select]
'''

# 按照abs和sum以后的最大值排列
A_mean = np.mean(np.abs(A), axis=0)
A_std = np.std(np.abs(A), axis=0)

mean_std = A_mean + 0.02 * 1 / A_std

index_A = np.argsort(- mean_std) # 从大到小排列
index_A = np.expand_dims(index_A, 0)
index_A = np.repeat(index_A, nb_select, axis=0)


# 连成长串输出
MUAPs_select = []
Prob_MUAPs_select = []

for j in range(nb_select):
    muap = []
    for i in range(A.shape[0]):
        muap = np.concatenate((muap, np.mean(A[:, index_A[i, j]]) * MUAPs[index_A[i, j]]), 0)
    MUAPs_select.append(muap)
    Prob_MUAPs_select.append(p_MUAPs[index_A[i, j]])

MUAPs_select = np.array(MUAPs_select)
Prob_MUAPs_select = np.array(Prob_MUAPs_select)


# 输出MUAPs
plt.figure()
plt.title('MUAPs')
for i in range(nb_select):
    ax = plt.subplot(nb_select, 1, i+1)
    plt.plot(MUAPs_select[i], color='orange')
    plt.ylim(-1,1)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
plt.show()

# 输出MUAPs的firing time
sys.path.append('E:\swt\Spike_Sorting')
from Spike_Sorting import spike_sorting
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

def plot_MUAPs_threhold(MUAPs, prob_MUAPs, threshold, nb_select=10, plot=True, sort_by_prob=True):
    '''
    按照阈值来对MUAPs进行spike sorting，并且输出结果
    MUAPs: n x data_dims
    threshold: 阈值
    nb_select: 最多输出的个数
    '''
    nb_select = min(MUAPs.shape[0], nb_select)


    spike_dict_list = []
    index_select = []
    for i in range(nb_select):
        if np.any(MUAPs[i] > threshold):
            index_select.append(i)
            spi = spike_sorting_intervals(MUAPs[i], threshold=threshold, overlap=10, plot=False)
            spi_dict = spi._get_dict()
            spike_dict_list.append(spi_dict)

    spike_dict_list = np.array(spike_dict_list)
    MUAPs = MUAPs[index_select]
    prob_MUAPs = prob_MUAPs[index_select]


    if sort_by_prob:
        MUAPs = MUAPs[np.argsort(-prob_MUAPs)]
        spike_dict_list = spike_dict_list[np.argsort(-prob_MUAPs)]

    if plot:
        plt.figure()
        for i, spi_dict in enumerate(spike_dict_list):
            spike_times = []
            ax = plt.subplot(len(spike_dict_list), 1, i+1)
            if len(spi_dict) > 0:
                for spike in spi_dict.keys():
                    spike_times.append(spi_dict[spike][2])
                plt.plot(MUAPs[i], color='orange')
                plt.vlines(spike_times, 0, 1)    
                plt.xlim(0, MUAPs.shape[1])
                plt.ylim(-1, 1)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xticks([])
                plt.yticks([])

        #plt.show()
    return 

plot_MUAPs_threhold(MUAPs_select, Prob_MUAPs_select, 0.05, 20, True, True)
plt.show()

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

remain_EMG = original_EMG - reconstruct_EMG
plt.figure()
plt.plot(remain_EMG + 0.5, '-.', c='c' ,label='Remaining EMG')
plt.ylim(0, 1)


# 输出混合矩阵A
plt.figure()
sns.set()
#A = A[sorted_index_A]
A = np.reshape(A, (-1, 1))
#ax = sns.heatmap(A, -0.5, 0.5)
ax = sns.heatmap(A)
plt.show()




