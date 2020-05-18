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

def train(x,gt):
    model.train()
    x = Variable(x)
    gt = Variable(gt)

    if args.cuda:
        x, gt = x.cuda(), gt.cuda()

    optimizer.zero_grad()
    output = model(x)
    output = torch.squeeze(output,1)
    
    '''print(output)
    print(gt)
    input()'''
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, gt)

    loss.backward()
    optimizer.step()

    return loss.item()
    

def val(x, gt):
    model.eval()
    x = Variable(x)
    gt = Variable(gt)

    if args.cuda:
        x, gt = x.cuda(), gt.cuda()

    output = model(x)
    output = torch.squeeze(output, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, gt)

    return loss.item(), output



def main():
    for epoch in range(1, args.epoch+1):
        total_loss=0.0
        tbar = tqdm.tqdm(TrainDataLoader, ncols=120)
        for batch_idx, (x, gt) in enumerate(tbar):
            loss = train(x, gt)
            total_loss += loss
            #print('Iter %d training loss = %.4f'%(batch_idx,loss))
            message = 'Training: Epoch: %d, loss=%f, total_loss=%f'%(epoch, loss, total_loss)
            tbar.set_description(message)

        total_loss = 0.0
        tbar_2 = tqdm.tqdm(ValDataLoader, ncols=120)
        predict = []
        for batch_idx, (x, gt) in enumerate(tbar_2):
            loss, pred = val(x, gt) 
            predict.append(pred)
            total_loss += loss
            message = 'Validating: Epoch: %d, loss=%f, total_loss=%f'%(epoch, loss, total_loss)
            tbar_2.set_description(message)
        
        if epoch%2 == 0:
            torch.save(model.state_dict(),'%snet_%03ds.pth' % (args.savemodel, epoch))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yaojy_cifar')
    parser.add_argument('--no-cuda',action='store_true',default=False, help='train with out cuda')
    parser.add_argument('--savemodel',default='./savemodel/',help='save model')
    parser.add_argument('--seed',default=1,type=int,help='random seed(default=1)')
    parser.add_argument('--dataset',default='./dataset/',help='dataset path')
    parser.add_argument('--epoch',default=300,type=int,help='epoch')

    args=parser.parse_args()
    args.cuda=not args.no_cuda and torch.cuda.is_available()
    print('CUDA = '+str(args.cuda))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed=args.seed
    data_file = args.dataset + 'train_data.txt'
    gt_file = args.dataset + 'train_truth.txt'

    TrainDataLoader = torch.utils.data.DataLoader(
        util.myTrainFolder(data_file, gt_file, training=True),
        batch_size= 16, shuffle= True, drop_last=False
    )
    ValDataLoader = torch.utils.data.DataLoader(
        util.myTrainFolder(data_file, gt_file, training=False),
        batch_size= 16, shuffle= False, drop_last=False
    )

    model = util.myNetwork()
    if args.cuda:
        model.cuda()
    #optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999))
    optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, lr=2*1e-3, momentum=0.9)
    main()
