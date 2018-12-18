import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys
sys.path.append('E:\swt\modified_DCGAN\EMG_model_WGAN_GP_with_filter')

import utilize

MUAPs = np.load('./S00201_MUAP_resample.npy')

threshold = 0.02
overlap = 10
nb_plots = 10

utilize.plot_MUAPs(nb_plots, MUAPs[:, :512])
utilize.cal_intervals_and_fit_with_gamma(MUAPs, threshold=threshold, overlap=overlap, plot=True)
utilize.plot_spike_time(nb_plots, MUAPs[:, :512], threshold, overlap, linewidth=3)


plt.show()