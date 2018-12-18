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
'''
这个new的分解方法是针对新的数据，mvc, open, close, rest 都是一大推的
'''
SAVE_DIR = './saveDecomposition_mvc'

RESTORE_PREVIOUS_MODEL = False
DEVICE = torch.device('cuda:3')
DTYPE = torch.float
DATA_DIMS = 512
GEN_SEARCH_NUM = 64
NOISE_DIMS = 100
NGF = 64
NDF = 64 

LEARNING_RATE_z = 0.002
LEARNING_RATE_A = 0.002
LAMBDA_1 = 0.1 # 用来控制到底是产生的数据真实和拟合的权重
LAMBDA_2 = 0.5

EPOCHS = 1000
SAVE_EVERY = 1000
PLOT_GRAD_EVERY = 100

DECREASE_LR = 100
UPDATE_A_EVERY = 1
UPDATE_z_EVERY = 10

EMG = np.load('./EMG_for_decomposition/trial2_data_dict.npy') # 需要分解的肌肉电信号
EMG = EMG.item()
EMG_mvc = copy.copy(EMG['mvc'])

for i, mvc in enumerate(EMG_mvc):
    EMG_mvc[i] = Filter.butter_highpass_filter(mvc, 100, 1000) # 高通滤波，截止频率100，采样率1000

EMG_mvc = torch.from_numpy(EMG_mvc)
EMG_mvc = EMG_mvc.to(DEVICE, DTYPE)
batch_size = EMG_mvc.shape[0]

# FIR滤波器
coeff_matrix = Filter.fir_filter_matrix(DATA_DIMS, 100, 1000)
coeff_matrix = torch.from_numpy(coeff_matrix)
coeff_matrix = coeff_matrix.to(DEVICE, DTYPE)

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

# 用来存储梯度grad的变化的
z_vel = torch.zeros_like(z)
A_vel = torch.zeros_like(A)

for epoch in range(EPOCHS):
    
    MUAPs = G_DC(z)

    MUAPs_logits = D_DC(MUAPs)
    MUAPs_logits = logits_func(MUAPs_logits)

    MUAPs = torch.squeeze(MUAPs)
    if GEN_SEARCH_NUM == 1:
        MUAPs = torch.unsqueeze(MUAPs, 0)

    MUAPs = torch.matmul(MUAPs, coeff_matrix) # 对每个MUAPs进行100Hz的高通滤波

    reconstruct_EMG = torch.matmul(A, MUAPs) #
    penalty_A = l1loss(A - torch.mean(A, 0), torch.zeros_like(A))
    #reconstruct_EMG = reconstruct_EMG.repeat((batch_size, 1))
    
    #loss = mseloss(reconstruct_EMG, EMG_mvc) + LAMBDA_1 * torch.mean(MUAPs_logits) + LAMBDA_2 * penalty_A
    loss = l1loss(reconstruct_EMG, EMG_mvc) + + LAMBDA_1 * torch.mean(MUAPs_logits)
    loss.backward()

    # 减小learning rate
    '''
    if (epoch + 1) > 3000:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
    elif (epoch + 1) > 3500:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
    elif (epoch + 1) > 4000:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
    '''

    epsilon = 0.000001

    with torch.no_grad():

        if epoch % UPDATE_z_EVERY == 0:
            z_grad = LEARNING_RATE_z * z.grad / (z.grad.data.norm(2) + epsilon)
            z -= z_grad + 0.5 * z_vel
            z_vel.data = z_grad.data

        # A 的更新频率要慢一些

        if epoch % UPDATE_A_EVERY == 0:
            A_grad = LEARNING_RATE_A * A.grad / (A.grad.data.norm(2) + epsilon)
            A -= A_grad + 0.5 * A_vel
            A_vel.data = A_grad.data

        # 把grad的值显示出来，方便调试            
        if (epoch + 1) % PLOT_GRAD_EVERY == 0:
            print('z grad:', torch.mean(torch.abs(z.grad / (z.grad.data.norm(2) + epsilon)  )).item())
            print('A grad:', torch.mean(torch.abs(A.grad / (A.grad.data.norm(2) + epsilon) )).item())
            print('Epoch: ', epoch, 'Loss: ', loss.item())

    z.grad.zero_()
    A.grad.zero_()

    if epoch % SAVE_EVERY == 0:
        np.save(os.path.join(SAVE_DIR, 'original_EMG.npy'), EMG_mvc.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'reconstruct_EMG.npy'), reconstruct_EMG.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'MUAPs.npy'), MUAPs.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'z.npy'), z.cpu().data.numpy())
        np.save(os.path.join(SAVE_DIR, 'A.npy'), A.cpu().data.numpy())

# 保存结果
results = {'z':z, 'A':A}
torch.save(results, os.path.join(SAVE_DIR, 'checkpoint.tar'))
