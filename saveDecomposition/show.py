import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio

nb_plots = 14

MUAPs = np.load('./MUAPs.npy')
A = np.squeeze(np.load('./A.npy'))

nb_plots = min(nb_plots, MUAPs.shape[0])

if nb_plots > 1:
	sorted_index_A = np.argsort(np.abs(A))[::-1] # A绝对值大的在前面
	A = A[sorted_index_A]
	MUAPs = MUAPs[sorted_index_A]

else:
	# 只有一个元素
	A = [A]
	MUAPs = [MUAPs]


#index_choose = np.random.choice(len(MUAPs), nb_plots)
#MUAPs = MUAPs[index_choose]

plt.figure()
# 输出MUAPs
for i in range(nb_plots):
    ax = plt.subplot(nb_plots, 1, i + 1)    
    plt.plot(np.squeeze(MUAPs[i] * A[i]))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])

# 对比original, reconstruct和 remain_EMG 
original_EMG = np.squeeze(np.load('original_EMG.npy'))
reconstruct_EMG = np.squeeze(np.load('reconstruct_EMG.npy'))

plt.figure()
plt.plot(original_EMG, label='EMG')
plt.plot(reconstruct_EMG, '--', label='Reconstructed EMG')
plt.ylim(-1, 1)
plt.legend(fontsize=15)
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
ax = sns.heatmap(A, -0.5, 0.5)
plt.show()