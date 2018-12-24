# -*- coding: utf-8 -*-
# 在生成的MUAPs部分引入一个FIR滤波器
# 因为我们使用的训练数据、sEMG，MUAPs都是经过100Hz高通滤波的数据
from __future__ import print_function, division
import numpy as np
import torch
import pdb
import Filter
import Model as Model
import matplotlib.pyplot as plt
import copy
import Optimizer_SGD as Optimizer

RESTORE_PREVIOUS_MODEL = True
DEVICE = torch.device('cuda')
DTYPE = torch.float
DATA_DIMS = 512
GEN_SEARCH_NUM = 64
NOISE_DIMS = 100
NGF = 64
NDF = 64

LEARNING_RATE_z = 0.0001
LEARNING_RATE_A = 0.0001
LAMBDA = 20 # 用来控制到底是产生的数据真实和拟合的权重

EPOCHS = 2000
SAVE_EVERY = 100
PLOT_GRAD_EVERY = 100

DECREASE_LR = 100
UPDATE_A_EVERY = 10
UPDATE_z_EVERY = 1

EMG = np.load('./EMG_for_decomposition/trial2_data_dict.npy') # 需要分解的肌肉电信号
EMG = EMG.item()
EMG = copy.copy(EMG['open'][2])

# FIR滤波器
coeff_matrix_np = Filter.fir_filter_matrix(DATA_DIMS, 100, 1000)
coeff_matrix = torch.from_numpy(coeff_matrix_np)
coeff_matrix = coeff_matrix.to(DEVICE, DTYPE)

EMG = np.matmul(EMG, coeff_matrix_np)

plt.plot(EMG)
plt.show()

EMG = np.expand_dims(EMG, axis=0)
EMG = torch.from_numpy(EMG)
EMG = EMG.to(DEVICE, DTYPE)


# 输入噪声z，混合矩阵A， 噪音noise， 考虑梯度
if RESTORE_PREVIOUS_MODEL:
    checkpoint = torch.load('./saveDecomposition_mvc/checkpoint.tar')
    z_ = checkpoint['z']
    z = torch.tensor(z_.data.cpu().numpy(), device=DEVICE, dtype=DTYPE, requires_grad=False)
    A = torch.randn(1, GEN_SEARCH_NUM, device=DEVICE, dtype=DTYPE, requires_grad=True)
else:
    z = torch.randn(GEN_SEARCH_NUM, NOISE_DIMS, 1, device=DEVICE, dtype=DTYPE, requires_grad=True)
    A = torch.randn(1, GEN_SEARCH_NUM, device=DEVICE, dtype=DTYPE, requires_grad=True)


G_DC = Model.build_dc_generator(ngf=NGF, noise_dims=NOISE_DIMS)
G_DC.load_state_dict(torch.load('./saveTorch/G_DC.pth'))
G_DC = G_DC.to(DEVICE, DTYPE)
G_DC.eval()

D_DC = Model.build_dc_classifier(ndf=NDF)
D_DC.load_state_dict(torch.load('./saveTorch/D_DC.pth'))
D_DC = D_DC.to(DEVICE, DTYPE)
D_DC.eval()

# 关闭对G_DC的梯度计算
for p in G_DC.parameters():
    p.requires_grad = False

for p in D_DC.parameters():
    p.requires_grad = False 


mseloss = torch.nn.MSELoss(size_average=False)
l1loss = torch.nn.L1Loss(size_average=False)

logits_func = torch.nn.Sigmoid()

optim_A = Optimizer.get_optimizer([A], learning_rate=LEARNING_RATE_A)
#optim_z = Optimizer.get_optimizer([z], learning_rate=LEARNING_RATE_z)

for epoch in range(EPOCHS):
    
    MUAPs = G_DC(z)
    MUAPs = torch.matmul(MUAPs, coeff_matrix) # 对每个MUAPs进行100Hz的高通滤波

    MUAPs_logits = D_DC(MUAPs)
    MUAPs_logits = logits_func(MUAPs_logits)

    MUAPs = torch.squeeze(MUAPs)
    if GEN_SEARCH_NUM == 1:
        MUAPs = torch.unsqueeze(MUAPs, 0)

    reconstruct_EMG = torch.mm(A, MUAPs) #

    loss = mseloss(reconstruct_EMG, EMG) + LAMBDA * torch.mean(MUAPs_logits)
    #loss = l1loss(reconstruct_EMG, EMG) + LAMBDA * torch.mean(MUAPs_logits)

    optim_A.zero_grad()
    #optim_z.zero_grad()
    loss.backward()

    if epoch % UPDATE_z_EVERY == 0:
        pass
        #optim_z.step()

    if epoch % UPDATE_A_EVERY == 0:
        optim_A.step()
        
    if (epoch + 1) % PLOT_GRAD_EVERY == 0:
        print('Epoch: ', epoch, 'Loss: ', loss.item())

    if epoch % SAVE_EVERY == 0:
        np.save('./saveDecomposition/original_EMG.npy', EMG.cpu().data.numpy())
        np.save('./saveDecomposition/reconstruct_EMG.npy', reconstruct_EMG.cpu().data.numpy())
        np.save('./saveDecomposition/MUAPs.npy', MUAPs.cpu().data.numpy())
        np.save('./saveDecomposition/z.npy', z.cpu().data.numpy())
        np.save('./saveDecomposition/A.npy', A.cpu().data.numpy())


# 保存结果
results = {'z':z, 'A':A}
torch.save(results, './saveDecomposition/checkpoint.tar')
