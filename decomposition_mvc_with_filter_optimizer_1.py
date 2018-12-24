# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import torch
import pdb
import Filter
import Model as Model
import matplotlib.pyplot as plt
import copy
import os
#import Optimizer_predictive_Adams as Optimizer
import Optimizer_SGD as Optimizer

'''
这个new的分解方法是针对新的数据，mvc, open, close, rest 都是一大推的
加入predictive adams，特别适合优化这种鞍点saddle point问题
_1 是为了改变参数，多试一试，方便比较结果
'''
SAVE_DIR = './saveDecomposition_mvc_1'

RESTORE_PREVIOUS_MODEL = True
DEVICE = torch.device('cuda:1')
DTYPE = torch.float
DATA_DIMS = 512
GEN_SEARCH_NUM = 140
NOISE_DIMS = 100
NGF = 64
NDF = 64 

LEARNING_RATE_z = 0.0001
LEARNING_RATE_A = 0.0001
LAMBDA_1 = 20 # 用来控制到底是产生的数据真实和拟合的权重
LAMBDA_2 = 0.5

EPOCHS = 100000
SAVE_EVERY = 1000
PLOT_GRAD_EVERY = 100

DECREASE_LR = 100
UPDATE_A_EVERY = 5
UPDATE_z_EVERY = 1

# FIR滤波器
coeff_matrix_np = Filter.fir_filter_matrix(DATA_DIMS, 100, 1000)
coeff_matrix = torch.from_numpy(coeff_matrix_np)
coeff_matrix = coeff_matrix.to(DEVICE, DTYPE)

# 读取数据
'''
EMG = np.load('./EMG_for_decomposition/trial2_data_dict.npy') # 需要分解的肌肉电信号
EMG = EMG.item()
EMG_mvc = copy.copy(EMG['mvc'])
EMG_mvc = EMG_mvc[:10] # 减小一点EMG_mvc的数目
'''

EMG = np.load('./EMG_for_decomposition/S002/S00201_resample.npy')
EMG = np.squeeze(EMG)
nb_EMG = 20
EMG = EMG[:nb_EMG * DATA_DIMS]
EMG_mvc = np.reshape(EMG, (nb_EMG, DATA_DIMS))

# 归一化 [-1,1]
EMG_mvc_max = np.max(EMG_mvc.flatten())
EMG_mvc_min = np.min(EMG_mvc.flatten())
EMG_mvc = (EMG_mvc - EMG_mvc_min) / (EMG_mvc_max - EMG_mvc_min) # [0, 1]
EMG_mvc = (EMG_mvc - 0.5) * 2

EMG_mvc = np.matmul(EMG_mvc, coeff_matrix_np)

EMG_mvc = torch.from_numpy(EMG_mvc)
EMG_mvc = EMG_mvc.to(DEVICE, DTYPE)
batch_size = EMG_mvc.shape[0]


# variable 输入噪声z，混合矩阵A， 考虑梯度
if RESTORE_PREVIOUS_MODEL:
    #checkpoint = torch.load(os.path.join(SAVE_DIR, 'checkpoints','checkpoint.tar'))
    checkpoint = torch.load(os.path.join(SAVE_DIR, 'checkpoint.tar'))
    z = checkpoint['z']
    A = checkpoint['A']
else:
    z = torch.randn(GEN_SEARCH_NUM, NOISE_DIMS, 1, device=DEVICE, dtype=DTYPE, requires_grad=True)
    A = torch.randn(batch_size, GEN_SEARCH_NUM, device=DEVICE, dtype=DTYPE, requires_grad=True)

G_DC = Model.build_dc_generator(ngf=NGF, noise_dims=NOISE_DIMS)
G_DC.load_state_dict(torch.load('./saveTorch/G_DC_66_6000.pth'))
G_DC = G_DC.to(DEVICE, DTYPE)
G_DC.eval()

D_DC = Model.build_dc_classifier(ndf=NDF)
D_DC.load_state_dict(torch.load('./saveTorch/D_DC_66_6000.pth'))
D_DC = D_DC.to(DEVICE, DTYPE)
D_DC.eval()

# 关闭对G_DC的梯度计算
for p in G_DC.parameters():
    p.requires_grad = False

for p in D_DC.parameters():
    p.requires_grad = False 

optim_A = Optimizer.get_optimizer([A], learning_rate=LEARNING_RATE_A)
optim_z = Optimizer.get_optimizer([z], learning_rate=LEARNING_RATE_z)

mseloss = torch.nn.MSELoss(size_average=False)
l1loss = torch.nn.L1Loss(size_average=False)

logits_func = torch.nn.Sigmoid()


for epoch in range(EPOCHS):
    
    MUAPs = G_DC(z)
    MUAPs = torch.matmul(MUAPs, coeff_matrix) # 对每个MUAPs进行100Hz的高通滤波
    MUAPs_logits = D_DC(MUAPs)
    MUAPs_logits = logits_func(MUAPs_logits)

    MUAPs = torch.squeeze(MUAPs)
    if GEN_SEARCH_NUM == 1:
        MUAPs = torch.unsqueeze(MUAPs, 0)

    reconstruct_EMG = torch.matmul(A, MUAPs) #
    penalty_A = l1loss(A - torch.mean(A, 0), torch.zeros_like(A))
    #reconstruct_EMG = reconstruct_EMG.repeat((batch_size, 1))
    
    #loss = mseloss(reconstruct_EMG, EMG_mvc) + LAMBDA_1 * torch.mean(MUAPs_logits) + LAMBDA_2 * penalty_A
    loss = mseloss(reconstruct_EMG, EMG_mvc) + LAMBDA_1 * torch.mean(1 - MUAPs_logits)
    #loss = l1loss(reconstruct_EMG, EMG_mvc) + + LAMBDA_1 * torch.mean(MUAPs_logits)

    optim_A.zero_grad()
    optim_z.zero_grad()
    loss.backward()

    if epoch % UPDATE_z_EVERY == 0:
        optim_z.step()

    if epoch % UPDATE_A_EVERY == 0:
        optim_A.step()

        # 把grad的值显示出来，方便调试            
    if (epoch + 1) % PLOT_GRAD_EVERY == 0:
        print('Epoch: ', epoch, 'Loss: ', loss.item())

    if epoch % SAVE_EVERY == 0:
        np.save(os.path.join(SAVE_DIR, 'original_EMG.npy'), EMG_mvc.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'reconstruct_EMG.npy'), reconstruct_EMG.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'MUAPs.npy'), MUAPs.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'z.npy'), z.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'A.npy'), A.cpu().data.numpy())

# 保存结果
results = {'z':z, 'A':A}
torch.save(results, os.path.join(SAVE_DIR, 'checkpoint.tar'))
