import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from spectral import SpectralNorm

# In[ ]:
# 数据采样率为1kHz

'''
Wasserstein GAN 中的 classifier 就是不采用Batch Norm的
'''

class build_dc_classifier(nn.Module):
    def __init__(self, ndf):
        super(build_dc_classifier, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, ndf, 2, 2, 0), # 1 * 512 => ndf * 256
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(ndf, 2*ndf, 2, 2, 0), # ndf * 256 => 2 x ndf * 128
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(2*ndf, 4*ndf, 2, 2, 0), # 2 x ndf * 128=> 4 x ndf * 64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(4*ndf, 8*ndf, 2, 2, 0), # 4 x ndf * 64 => 8 x ndf * 32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(8*ndf, 16*ndf, 2, 2, 0), # 8 x ndf * 32 => 16 x ndf * 16
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv1d(16*ndf, 32*ndf, 2, 2, 0), # 16 x ndf * 16 => 32 x ndf * 8
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32*ndf, 64*ndf, 2, 2, 0), # 32 x ndf * 8 => 64 x ndf * 4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64*ndf, 128*ndf, 2, 2, 0), # 64 x ndf * 4 => 128 x ndf * 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128*ndf, 1, 2, 2, 0), # 128 x ndf * 2 => 1 * 1 判断真假        
        )
        #self.fc = nn.Linear(128*ndf*2, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = x.view(x.shape[0], -1)
        #x = self.fc(x)
        return x


# 注意nn.ConvTranspose2d的计算是和Conv2d相反的，padding的时候要注意
# 
# 公式 $out=\frac{in-F+2P}{S} + 1$ 是计算conv2d的，利用convTranspose2d时要相反
# 
# F是kernel_size, P为padding，S为stride

# In[ ]:

class build_dc_generator(nn.Module): 
    def __init__(self, ngf, noise_dims):
        super(build_dc_generator, self).__init__()
        self.convTranspose1d = nn.Sequential(
            
            nn.ConvTranspose1d(noise_dims, 128 * ngf, 2, 2, 0), # noise_dims * 1 => 128 x ngf * 2
            nn.BatchNorm1d(128 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(128 * ngf, 64 * ngf, 2, 2, 0), # 128 x ngf * 2 => 64 x ngf * 4
            nn.BatchNorm1d(64 * ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64 * ngf, 32 * ngf, 2, 2, 0), # 64 x ngf * 4 => 32 x ngf * 8
            nn.BatchNorm1d(32 * ngf),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose1d(32 * ngf, 16 * ngf, 2, 2, 0), # 32 x ngf * 8 => 16 x ngf * 16
            nn.BatchNorm1d(16 * ngf),
            nn.ReLU(inplace=True),           

            nn.ConvTranspose1d(16 * ngf, 8 * ngf, 2, 2, 0), # 16 x ngf * 16 => 8 x ngf * 32
            nn.BatchNorm1d(8 * ngf),
            nn.ReLU(inplace=True),    

            nn.ConvTranspose1d(8 * ngf, 4 * ngf, 2, 2, 0), # 8 x ngf * 32 => 4 x ngf * 64
            nn.BatchNorm1d(4 * ngf),
            nn.ReLU(inplace=True),               
            
            nn.ConvTranspose1d(4 * ngf, 2 * ngf, 2, 2, 0), # 4 x ngf * 64 => 2 x ngf * 128
            nn.BatchNorm1d(2 * ngf),
            nn.ReLU(inplace=True),    

            nn.ConvTranspose1d(2 * ngf, 1 * ngf, 2, 2, 0), # 2 x ngf * 128 => 1 x ngf * 256
            nn.BatchNorm1d(1 * ngf),
            nn.ReLU(inplace=True),     
           
            nn.ConvTranspose1d(ngf, 1, 2, 2, 0), # ngf * 256 => 1 * 512 
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.convTranspose1d(x)
        return x