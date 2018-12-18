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

RESTORE_PREVIOUS_MODEL = True
DEVICE = torch.device('cuda')
DTYPE = torch.float
DATA_DIMS = 512
GEN_SEARCH_NUM = 64
NOISE_DIMS = 100
NGF = 64
NDF = 64

LEARNING_RATE_z = 0.001
LEARNING_RATE_A = 0.001
LAMBDA = 0.1 # 用来控制到底是产生的数据真实和拟合的权重

EPOCHS = 20000
SAVE_EVERY = 1000
PLOT_GRAD_EVERY = 100

DECREASE_LR = 100
UPDATE = 1

EMG = np.load('./EMG_for_decomposition/trial2_data_dict.npy') # 需要分解的肌肉电信号
EMG = EMG.item()
EMG = copy.copy(EMG['open'][2])

EMG = Filter.butter_highpass_filter(EMG, 100, 1000) # 高通滤波，截止频率100，采样率1000

plt.plot(EMG)
plt.show()

EMG = np.expand_dims(EMG, axis=0)
EMG = torch.from_numpy(EMG)
EMG = EMG.to(DEVICE, DTYPE)

# FIR滤波器
coeff_matrix = Filter.fir_filter_matrix(DATA_DIMS, 100, 1000)
coeff_matrix = torch.from_numpy(coeff_matrix)
coeff_matrix = coeff_matrix.to(DEVICE, DTYPE)

# 输入噪声z，混合矩阵A， 噪音noise， 考虑梯度
if RESTORE_PREVIOUS_MODEL:
    checkpoint = torch.load('./saveDecomposition/checkpoint.tar')
    z = checkpoint['z']
    A = checkpoint['A']
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

    reconstruct_EMG = torch.mm(A, MUAPs) #

    loss = mseloss(reconstruct_EMG, EMG) + LAMBDA * torch.mean(MUAPs_logits)
    #loss = l1loss(reconstruct_EMG, EMG) + LAMBDA * torch.mean(MUAPs_logits)

    loss.backward()

    # 减小learning rate
    '''
    if (epoch + 1) > 3000:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
        LEARNING_RATE_noise = LEARNING_RATE_noise * 0.1
    elif (epoch + 1) > 3500:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
        LEARNING_RATE_noise = LEARNING_RATE_noise * 0.1
    elif (epoch + 1) > 4000:
        LEARNING_RATE_A = LEARNING_RATE_A * 0.1
        LEARNING_RATE_noise = LEARNING_RATE_noise * 0.1
    '''

    epsilon = 0.000001

    with torch.no_grad():

        z_grad = LEARNING_RATE_z * z.grad / (z.grad.data.norm(2) + epsilon)
        z -= z_grad + 0.5 * z_vel
        z_vel.data = z_grad.data

        # A 的更新频率要慢一些

        if epoch % UPDATE == 0:
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
        np.save('./saveDecomposition/original_EMG.npy', EMG.cpu().data.numpy())
        np.save('./saveDecomposition/reconstruct_EMG.npy', reconstruct_EMG.cpu().data.numpy())
        np.save('./saveDecomposition/MUAPs.npy', MUAPs.cpu().data.numpy())
        np.save('./saveDecomposition/z.npy', z.cpu().data.numpy())
        np.save('./saveDecomposition/A.npy', A.cpu().data.numpy())


# 保存结果
results = {'z':z, 'A':A}
torch.save(results, './saveDecomposition/checkpoint.tar')

'''
结果统计： 
mse lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=256  loss=24.354
mse lr=0.01, A=0.001,  noise=0.001, GEN_SEARCH_NUM=256    loss=9.6972
l1  lr=0.01, A=0.001,  noise=0.001, GEN_SEARCH_NUM=256    loss=102.08
l1  lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=256   loss=71.7288
l1  lr=0.01, A=0.00001, noise=0.00001, GEN_SEARCH_NUM=256   loss=71629.7
l1  lr=0.01, A=0.01, noise=0.01, GEN_SEARCH_NUM=256   loss= 2550/490之间反复震荡
l1  lr=0.01, A=0.0003, noise=0.0003, GEN_SEARCH_NUM=256 loss= 200/500之间反复震荡
l1  lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=256   loss=34.069 3000epochs 之后降低为 0.1
l1  lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=256   loss=38.6329 3000epochs 之后降低为0.5倍 4000eopchs再次降低为0.5倍
l1  lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=16   loss=35.655
l1  lr=0.01, A=0.0001, noise=0.0001, GEN_SEARCH_NUM=16   loss=33.4626      10000epochs
l1  lr=0.01, A=0.0002, UPDATE_A=2, noise=0.0002, GEN_SEARCH_NUM=16   loss=32.7      10000epochs
l1  lr=0.01, A=0.001, UPDATE_A=2, noise=0.001, GEN_SEARCH_NUM=16   loss=42      10000epochs
l1  lr=0.001, A=0.001, UPDATE_A=2, noise=0.001, GEN_SEARCH_NUM=16   loss=48.0407 10000epochs
经过实验发现，必须得有A啊，对降低误差作用很大
'''