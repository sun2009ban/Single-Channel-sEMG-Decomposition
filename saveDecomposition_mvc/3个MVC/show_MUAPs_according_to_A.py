import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('E:\swt\modified_DCGAN\EMG_model_WGAN_GP_with_filter')

import scipy.stats as stats
import utilize

'''
这个主要是把这么多的MUAPs和A排列成spikes的形式
'''

nb_select = 20 # 只显示最大的几个值

MUAPs = np.load('./MUAPs.npy')
A = np.squeeze(np.load('./A.npy'))

print('MUAPs: ', MUAPs.shape)
print('A: ', A.shape)

nb_seg = A.shape[0]

for i in range(nb_seg):
	muaps, a = utilize.find_top_n_MUAPs_according_to_A(A[i, :], MUAPs, 0.1, nb_select)
	utilize.plot_MUAPs(nb_select, muaps, a)
	utilize.plot_spike_time(nb_select, muaps, threshold=0.1, overlap=10)
plt.show()


original_EMG = np.load('original_EMG.npy')
approximated_EMG = np.load('reconstruct_EMG.npy')

utilize.plot_EMG_and_its_approximation(original_EMG, approximated_EMG)
plt.show()