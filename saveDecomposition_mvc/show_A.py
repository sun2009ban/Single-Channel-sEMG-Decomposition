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

nb_select = 50 # 只显示最大的10个值

MUAPs = np.load('./MUAPs.npy')
A = np.squeeze(np.load('./A.npy'))
z = np.squeeze(np.load('./z.npy'))

'''
# 每一段都按照自己最大的排列，直接粘合
index_A = np.argsort(-np.abs(A), axis=1) # 加上负号是为了使其从大到小排列
index_A = index_A[:, :nb_select]
'''

# 按照abs和sum以后的最大值排列
index_A = np.argsort(-np.sum(np.abs(A), axis=0))
index_A = np.expand_dims(index_A, 0)
index_A = np.repeat(index_A, nb_select, axis=0)

# 连成长串输出
MUAPs_select = []
for j in range(nb_select):
    muap = []
    for i in range(A.shape[0]):
        muap = np.concatenate((muap, MUAPs[index_A[i, j]]), 0)
    MUAPs_select.append(muap)

MUAPs_select = np.array(MUAPs_select)

# 输出MUAPs
for i in range(nb_select):
    plt.subplot(nb_select, 1, i+1)
    plt.plot(MUAPs_select[i])
plt.show()


'''
转换思路，利用KMeans来解决问题
'''
# 对A按照列进行分类，相同类的叠加放在一起
nb_clusters = 7
from sklearn.cluster import KMeans
A_T = np.transpose(A)
kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(A_T)
A_T_labels = kmeans.labels_ 



MUAPs_clusters = np.zeros((nb_clusters, MUAPs.shape[1]))

for i, label in enumerate(A_T_labels):
    MUAPs_clusters[label] += MUAPs[i]

# 输出MUAPs_clusters
plt.figure()
for i in range(nb_clusters):
    plt.subplot(nb_clusters, 1, i+1)
    plt.plot(MUAPs_clusters[i])
plt.show()


exit()

# 输出A
plt.figure()
plt.title('A')
for i in range(A.shape[0]):
    plt.bar(x=np.arange(A.shape[1]) + 0.045 * i, height=np.abs(A[i]), width=0.04, label=str(i))
plt.legend()

np.save('./MUAPs.npy', MUAPs) # 用sorted的MUAPs替换原来的

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




