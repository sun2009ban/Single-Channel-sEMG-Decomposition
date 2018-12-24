import numpy as np
import matplotlib.pyplot as plt
import pdb
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid) # 使得对所有元素可用

data = np.load('./real_data_dict.npy').item()

fake_data = np.squeeze(data['real_data'])
scores = np.squeeze(data['scores'])


scores = sigmoid_v(scores)

nb_plots = 32

# 输出生成的MUAPTs
plt.figure()
for i in range(nb_plots):
    ax = plt.subplot(nb_plots, 1, i + 1)    
    plt.plot(fake_data[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])


# 输出对应的probability
plt.figure()
x = np.arange(nb_plots) + 1
plt.barh(x, scores[:nb_plots])


plt.show()

