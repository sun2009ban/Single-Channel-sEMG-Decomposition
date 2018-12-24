import random
import numpy as np


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

        self.position = 0

    def push(self, data):
        """保存 data"""
        if len(self.memory) < self.capacity:
            self.memory.append(None) # 加入一个None占位，这样下面才㴰按照位置插入
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_memory():
    return ReplayMemory(2000) # 存储的batch的数目