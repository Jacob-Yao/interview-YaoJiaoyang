import os
import struct
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

## Network structure
class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork,self).__init__()
        self.fc = nn.Sequential(nn.Linear(3,64),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(64,16),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(16,1))

    def forward(self, data_input):
        out = self.fc(data_input)
        return out

## Data folders
class myTrainFolder(torch.utils.data.Dataset):
    def __init__(self, data_file, gt_file, training=True):
        dataset = self._read_data(data_file)
        gt = self._read_gt(gt_file)
        n = dataset.shape[0]
        assert n==gt.shape[0]
        if training:
            self.dataset = dataset[:int(n*0.8),:]
            self.gt = gt[:int(n*0.8)]
        else:
            self.dataset = dataset[int(n*0.8):,:]
            self.gt = gt[int(n*0.8):]

    def _read_data(self, file_name):
        with open(file_name, 'r') as f1:
            content = f1.read()
            content = content.split('\n')[1:-1]
            content = ('\t'.join(content)).split('\t')
            out = np.array([float(a) for a in content])
            return out.reshape([-1, 3])

    def _read_gt(self, file_name):
        with open(file_name, 'r') as f1:
            content = f1.read()
            content = content.split('\n')[1:-1]
            content = ('\t'.join(content)).split('\t')
            out = np.array([float(a) for a in content])
            return out

    def __getitem__(self, index):
        data = torch.from_numpy(self.dataset[index]).type(torch.float)
        gt = torch.tensor(self.gt[index]).type(torch.float)
        return data, gt
    
    def __len__(self):
        return self.dataset.shape[0]


class myTestFolder(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.dataset = self._read_data(data_file)

    def _read_data(self, file_name):
        with open(file_name, 'r') as f1:
            content = f1.read()
            content = content.split('\n')[1:-1]
            content = ('\t'.join(content)).split('\t')
            out = np.array([float(a) for a in content])
            return out.reshape([-1, 3])
            
    def __getitem__(self, index):
        data = torch.from_numpy(self.dataset[index]).type(torch.float)
        return data
    
    def __len__(self):
        return self.dataset.shape[0]

## result writer
def write_result(file_name, result):
    with open(file_name, 'w') as f1:
        f1.write('y\n')
        for row in range(len(result)):
            f1.write(str(result[row]) + '\n')