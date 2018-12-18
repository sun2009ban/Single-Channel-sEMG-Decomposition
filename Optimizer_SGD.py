import numpy as np
import torch
import math
from torch import optim

# In[ ]:

# 为了和Spectral Normalization 联用
def get_optimizer(list_of_parameters, learning_rate, momentum=0.5):
    optimizer =optim.SGD(list_of_parameters, lr=learning_rate, momentum=momentum)
    return optimizer
