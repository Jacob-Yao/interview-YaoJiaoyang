import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm as tqdm 

import util



def test(x):
    model.eval()
    x = Variable(x)

    if args.cuda:
        x = x.cuda()

    output = model(x)
    output = torch.squeeze(output, 1)
    
    return output.item()



def main():
    tbar_2 = tqdm.tqdm(TestDataLoader, ncols=120)
    predict = []
    for batch_idx, (x) in enumerate(tbar_2):
        pred = test(x) 
        predict.append(pred)
        message = 'Testing: '
        tbar_2.set_description(message)
    util.write_result('test_predicted.txt', predict)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yaojy_cifar')
    parser.add_argument('--no-cuda',action='store_true',default=False, help='train with out cuda')
    parser.add_argument('--savemodel',default='./savemodel/net_074s.pth',help='save model')
    parser.add_argument('--seed',default=1,type=int,help='random seed(default=1)')
    parser.add_argument('--dataset',default='./dataset/',help='dataset path')
    parser.add_argument('--epoch',default=300,type=int,help='epoch')

    args=parser.parse_args()
    args.cuda=not args.no_cuda and torch.cuda.is_available()
    print('CUDA = '+str(args.cuda))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed=args.seed
    data_file = args.dataset + 'test_data.txt'

    TestDataLoader = torch.utils.data.DataLoader(
        util.myTestFolder(data_file),
        batch_size= 1, shuffle= False, drop_last=False
    )

    model = util.myNetwork()
    model.load_state_dict(torch.load(args.savemodel))
    if args.cuda:
        model.cuda()
    main()
