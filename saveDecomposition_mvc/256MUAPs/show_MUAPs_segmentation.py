import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('E:\swt\modified_DCGAN\EMG_model_WGAN_GP_with_filter')

import utilize
import scipy.stats as stats

'''
按照A的一段一段来看MUAPs，最后再重叠
'''

nb_select = 50 # 只显示最大的几个值

MUAPs = np.load('./h_c.npy')
A = np.squeeze(np.load('./A_c.npy'))
p_MUAPs = np.load('./p_c.npy')

print('MUAPs: ', MUAPs.shape)
print('A: ', A.shape)
print('MUAPs Probability: ', p_MUAPs.shape)

nb_seg = A.shape[0]

for i in range(nb_seg):
	muaps, a = utilize.find_top_n_MUAPs_according_to_A(A[i, :], MUAPs, 0.5, nb_select)
	utilize.plot_MUAPs(nb_select, muaps, a)
	utilize.plot_spike_time(nb_select, MUAPs, threshold=0.1, overlap=10)
plt.show()