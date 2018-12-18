from __future__ import print_function, division
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pdb
import Filter
import matplotlib.pyplot as plt
'''
新的读取数据的
'''

FILE_EXTENSIONS = [".npy"]

def is_file(filename):
    '''
    判断filename是否是以FILE_EXTENSIONS中的为结尾
    '''
    return any(filename.endswith(extension) for extension in FILE_EXTENSIONS)

def walk_through_dir(directory):
    '''
    遍历目录dir下面的以FILE_EXTENSIONS为结尾的文件
    返回值为文件的路径
    '''
    file_path = []

    for root, _, fnames in sorted(os.walk(directory)):
        for fname in sorted(fnames):
            if is_file(fname):
                path = os.path.join(directory, fname) #把目录和
                file_path.append(path)

    return file_path

class EMG(Dataset):
    def __init__(self, data_dir, out_len, out_overlap=0):
        self.out_len = out_len
        self.gen_inputs(data_dir, out_len, out_overlap)

    def cut_sequence_to_matrix(self, sequence, cut_len, overlap):
        '''
        sequence: 输入序列 numpy vector
        cut_len: 需要分的长度
        输出：把sequence按照cut_len的长度分成段，放在一个list中
        '''
        assert len(sequence) > cut_len
        assert cut_len > overlap
        
        # 数据变换为[-1,1]之间
        seq_max = np.amax(sequence)
        seq_min = np.amin(sequence)
        sequence = (sequence - seq_min) / (seq_max - seq_min) # [0, 1]
        sequence = 2 * (sequence - 0.5) # [-1, 1]

        assert np.all(sequence >= -1)
        assert np.all(sequence <= 1) 
        
        gen_seq = []

        # 开头需要单独拿出来
        seq = sequence[0 : cut_len]
        seq = np.expand_dims(seq, 0)
        gen_seq.append(seq)

        i = 1
        while i * cut_len - overlap + cut_len < len(sequence):
            seq = sequence[i * cut_len - overlap : i * cut_len - overlap + cut_len]
            seq = np.expand_dims(seq, 0)
            gen_seq.append(seq)
            i += 1

        return np.vstack(gen_seq)

    def del_emg_too_small(self, emg):
        '''
        截断后的emg矩阵里面，有个别的emg信号非常的小，
        因此把这些emg信号删除掉
        '''
        del_index = []
        for i in range(emg.shape[0]):
            if emg[i].max() - emg[i].min() < 0.1: # 最大值和最小值差值小于0.1，就认为这个里面啥都没有
                del_index.append(i)
        emg = np.delete(emg, del_index, 0) #把对应的行删除
        return emg

    def get_min_max_from_seq(self, seq):
        return seq.min(), seq.max()

    def filter(self, raw_data, cut_off=100):    
        '''100Hz High-Pass Filter'''
        fs = 1000
        filtered_data = Filter.butter_highpass_filter(raw_data, cut_off, fs, order=5)
        return filtered_data


    def gen_inputs(self, dir_path, out_len, out_overlap=0):
        '''
        dir_path: 采样率均为1kHz，读取目录下面的全部结尾为.npy的文件
        out_len: 输出的iEMG长度
        '''
        '''读取iEMG'''
        EMG = []
        for path in walk_through_dir(dir_path):
            print('MUAPTs file path: ', path)
            seq_data = np.load(path)
            seq_data = np.squeeze(seq_data)
            seq_data = self.filter(seq_data) #对数据进行高通滤波
            #plt.figure()
            #plt.plot(seq_data)
            print(path, 'Seq Length: ', len(seq_data))
            EMG.append(self.cut_sequence_to_matrix(seq_data, out_len, out_overlap))
        EMG = np.concatenate(EMG, axis=0)

        EMG = self.del_emg_too_small(EMG)

        self.x = EMG
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        x = np.reshape(x, (1, self.out_len)) # (Channel, Length) for conv1d
        return torch.Tensor(x)

def get_train_data(data_dims, batch_size, nb_repeat=1000, data_dir='./EMG_npy/sorted'):
    EMG_set = []
    for _ in range(nb_repeat): # 这个是决定了按照overlap取出多少个数据
        EMG_set.append(EMG(data_dir, data_dims, out_overlap=int(512 * np.random.rand())))

    train_set = torch.utils.data.ConcatDataset(EMG_set)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print('Finish Loading Data!')
    return train_data

if __name__ == '__main__':
    train_data = get_train_data(512, 64)
    data = []
    for batch_data in train_data:
        data.append(batch_data.data.numpy())

    data = np.concatenate(data)
    data = np.squeeze(data)

    np.save('./results/real_data.npy', data)