import numpy as np
import torch
import math
from torch import optim

# In[ ]:

# 普通adam优化器
def get_optimizer(net, learning_rate, betas=(0.9, 0.99)):
    parameters = []
    for p in net.parameters():
        if p.requires_grad:
            parameters.append(p)
    optimizer =optim.Adam(parameters, lr=learning_rate, betas=betas)
    return optimizer
