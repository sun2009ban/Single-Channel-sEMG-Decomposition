'''
这个对应论文里面的MUAPTs combination
'''
import torch
import numpy as np
import Model as Model
import os
import pdb
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid) # 使得对所有元素可用

DATA_DIMS = 512
NDF = 64
DEVICE = torch.device("cuda:0") # 指定显卡号
DTYPE = torch.float
SAVE_DIR = './saveTorch'

# 读取数据
OUTPUT_DIR = './saveDecomposition_mvc'

# 原始的第一次
#MUAPs = np.load(os.path.join(OUTPUT_DIR, 'MUAPs.npy'))
#A = np.load(os.path.join(OUTPUT_DIR, 'A.npy'))
# 接下来的
MUAPs = np.load(os.path.join(OUTPUT_DIR, 'h_c.npy'))
A = np.load(os.path.join(OUTPUT_DIR, 'A_c.npy'))


D_DC = Model.build_dc_classifier(ndf=NDF).to(DEVICE, DTYPE)
D_DC.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'D_DC_94_6000.pth')))

batch_size = MUAPs.shape[0]

MUAPs = torch.from_numpy(MUAPs)
MUAPs = MUAPs.to(DEVICE, DTYPE)
MUAPs = torch.unsqueeze(MUAPs, 1)
prob_MUAPs = sigmoid_v(D_DC(MUAPs).data.cpu().numpy())
prob_MUAPs = np.squeeze(prob_MUAPs)

c = []
prob_combine_MUAPs = []
for i in range(batch_size - 1):
	for j in range(i + 1, batch_size):
		combine_MUAPs = MUAPs[i] + MUAPs[j]
		combine_MUAPs = torch.unsqueeze(combine_MUAPs, 0)
		prob_combine = sigmoid_v(D_DC(combine_MUAPs).data.cpu().numpy())
		
		if prob_combine[0, 0] > max(prob_MUAPs[i], prob_MUAPs[j]): #and np.sum(np.abs(A[:, i] - A[:,j])) < 1:
			c.append((i,j))
			prob_combine_MUAPs.append(prob_combine[0])

MUAPs = MUAPs.data.cpu().numpy() # tensor => numpy
prob_combine_MUAPs = np.array(prob_combine_MUAPs).flatten() 
h_c = []
A_c = []
p_c = []  # combine之后的probability of being real
c_used = []
sort_index = np.argsort(-prob_combine_MUAPs)

for k in sort_index:
	i, j = c[k]
	if i not in c_used and j not in c_used:
		A_c.append(0.5 * (A[:, i] + A[:, j]))		
		h_c.append(MUAPs[i] + MUAPs[j])
		p_c.append(prob_combine_MUAPs[k])

		c_used.append(i)
		c_used.append(j)

for i in range(batch_size):
	if i not in c_used:
		h_c.append(MUAPs[i])
		A_c.append(A[:, i])
		p_c.append(prob_MUAPs[i])

A_c = np.array(A_c)
h_c = np.squeeze(np.array(h_c))
p_c = np.squeeze(np.array(p_c)) 

print('A_c shape', A_c.shape)
print('h_c shape', h_c.shape)
print('p_c.shape', p_c.shape)

np.save(os.path.join(OUTPUT_DIR, 'A_c.npy'), np.transpose(A_c))
np.save(os.path.join(OUTPUT_DIR, 'h_c.npy'), h_c)
np.save(os.path.join(OUTPUT_DIR, 'p_c.npy'), p_c)