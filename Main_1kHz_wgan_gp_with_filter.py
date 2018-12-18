# -*- coding: utf-8 -*-
# 希望构建标准模型，每次替换一点点就可以了
'''
这个模型里面是 Wasserstein GAN with gradient penalty
所谓的with filter的意思是discriminator 和 generator的输出的波形会经过一个100Hz的高通滤波器
注意这里面的数据都是[-1, 1]之间的
'''
from __future__ import print_function, division
import torch
from torch import nn
import pdb 
import numpy as np
import os

import Data as Data 
import Model as Model
import Loss as Loss
import Optimizer as Optimizer
import Memory
import Train_gp_with_filter as Train # gradient penalty
import Filter

# 参数
# x是512这么长的序列
DATA_DIMS = 512
BATCH_SIZE = 32 #文章中表示batch_size增大有助于提高稳定性
NUM_EPOCHS = 1000
NOISE_DIMS = 100
NDF = 64
NGF = 64 #32
RESTORE_PREVIOUS_MODEL = False
DATA_DIR = './EMG_npy/sorted'
SAVE_DIR = './WGAN_GP_with_filter_saveTorch'
SAVE_OUTPUT_DIR = './saveOutput'
LEARNING_RATE_D = 1e-4
LEARNING_RATE_G = 1e-4

# 采用下面这种模式保证了不会总用同一个正确样本训练DIS
D_EVERY = 1
G_EVERY = 3
SAVE_EVERY = 5000
SHOW_EVERY = 1000

# 决定是否在real_data上加入噪音 NOISE_AMP * N(0, I)
# 加入noise对网络稳定性的影响很大的，加入大的noise直接就稳定了
ADD_NOISE = True
NOISE_AMP = 0.01

# Discriminator 梯度的惩罚系数，主要靠这个来改变鞍点的稳定性了
GRADIENT_PENALTY = 5

# 是否裁剪梯度
GRADIENT_CLIP = False

# 生成数据
GEN_SEARCH_NUM = 512
GEN_NUM = 32

DEVICE = torch.device("cuda") # 指定显卡号
DTYPE = torch.float
 
USE_MEMORY = False

# 读取数据
train_dataset = Data.get_train_data(DATA_DIMS, BATCH_SIZE, 10, DATA_DIR)

# 建立模型

D_DC = Model.build_dc_classifier(ndf=NDF).to(DEVICE, DTYPE)
G_DC = Model.build_dc_generator(ngf=NGF, noise_dims=NOISE_DIMS).to(DEVICE, DTYPE)

# 读取存储的模型
if RESTORE_PREVIOUS_MODEL:
    #D_DC.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'new_start', 'D_DC_207_0.pth')))
    #G_DC.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'new_start', 'G_DC_207_0.pth')))
    D_DC.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'D_DC_175_0.pth'), map_location='cuda'))
    G_DC.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'G_DC_175_0.pth'), map_location='cuda'))


# 建立优化器
D_DC_optim = Optimizer.get_optimizer(D_DC, LEARNING_RATE_D)
G_DC_optim = Optimizer.get_optimizer(G_DC, LEARNING_RATE_G)

#建立记忆
MEMORY = Memory.get_memory()

# FIR滤波器
coeff_matrix = Filter.fir_filter_matrix(DATA_DIMS, 100, 1000)
coeff_matrix = torch.from_numpy(coeff_matrix)
coeff_matrix = coeff_matrix.to(DEVICE, DTYPE)

# 训练模型
Train.train_dc_gan(train_dataset, D_DC, G_DC, D_DC_optim, G_DC_optim, Loss.discriminator_loss, Loss.generator_loss, filter=coeff_matrix, show_every=SHOW_EVERY, noise_size=NOISE_DIMS, gaussian_noise=False, num_epochs=NUM_EPOCHS,
    d_every=D_EVERY, g_every=G_EVERY, save_every=SAVE_EVERY, add_noise=ADD_NOISE, noise_amp=NOISE_AMP, gradient_penalty=GRADIENT_PENALTY, use_memory=USE_MEMORY, memory=MEMORY, gradient_clip=GRADIENT_CLIP, device=DEVICE,
    dtype=DTYPE, save_dir=SAVE_DIR, save_output_dir=SAVE_OUTPUT_DIR, use_visdom=False)


# 存储模型
torch.save(D_DC.state_dict(), os.path.join(SAVE_DIR, 'D_DC.pth'))
torch.save(G_DC.state_dict(), os.path.join(SAVE_DIR, 'G_DC.pth'))

# 任意输出64个由噪音产生的数据
noise = torch.randn(GEN_SEARCH_NUM, NOISE_DIMS, 1)
noise = noise.to(DEVICE, DTYPE)
fake_data = G_DC(noise)
fake_data = torch.matmul(fake_data, coeff_matrix) 
scores = D_DC(fake_data).data

# 挑选好的
index = torch.topk(scores, GEN_NUM, dim=0)[1] #返回值是tuple，里面两个1维度的Tensor, 分别对应值和位置，这里只取了位置，因此[1]
results = []
for i in index:
    results.append(fake_data.data[i])

results = torch.stack(results)
results = torch.squeeze(results)

# 保存结果为ndarray
results_np = results.cpu().data.numpy()
np.save(os.path.join('./results', 'result.npy'), results_np)

# 保存fake的结果
fake_data_np = fake_data.data.cpu().numpy()
fake_data_scores_np = scores.cpu().numpy()
np.save(os.path.join('./results', 'fake_data_dict.npy'), {'fake_data':fake_data_np, 'scores':fake_data_scores_np})

# 保存real的结果
real_data_np = np.load('./results/real_data.npy')
real_data_np = real_data_np[:BATCH_SIZE]
real_data = torch.from_numpy(np.expand_dims(real_data_np, 1))
real_data = real_data.to(DEVICE)
real_data = torch.matmul(real_data, coeff_matrix)
real_data_scores = D_DC(real_data)
real_data_scores_np = real_data_scores.data.cpu().numpy()
np.save(os.path.join('./results', 'real_data_dict.npy'), {'real_data':real_data_np, 'scores':real_data_scores_np})