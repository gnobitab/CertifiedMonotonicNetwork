### simple 2-d case with PyTorch
### monotonic: capital-gain, weekly hours of work and education level, and the gender wage gap
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.networks import *
import torchvision.datasets as dset
import torchattacks

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument("-f", "--freezefeature", help="freeze resnet18", action="store_true")
args = parser.parse_args()

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

train_set = dset.MNIST(root='/home/xcliu/datasets', train=True, transform=trans)
idx = (train_set.targets==2) | (train_set.targets==7)
train_set.targets = train_set.targets[idx]
train_set.data = train_set.data[idx]

test_set = dset.MNIST(root='/home/xcliu/datasets', train=False, transform=trans)
idx = (test_set.targets==2) | (test_set.targets==7)
test_set.targets = test_set.targets[idx]
test_set.data = test_set.data[idx]

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

### Full-batch-training
criterion = nn.CrossEntropyLoss()

net = InterConv_Robust(layer_num=1, hidden_num=50, class_num=2)

param_amount = 0
for p in net.named_parameters():
    if not 'resnet18' in p[0]:
        print(p[0])
        param_amount += p[1].numel()
print('total param amount:', param_amount)

net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

atk = torchattacks.PGD(net, eps = 32/255, alpha = 2/255, steps=30)
def test():            
    correct_num = 0.
    label_num = 0.
    
    for batch_idx, (x, target) in enumerate(test_loader):
        x = x.cuda()
        target = target.cuda()
        target[target==2] = 0
        target[target==7] = 1
        adv_x = atk(x, target)
        
        out = net(adv_x)
        _, pred_label = torch.max(out.data, 1)
        correct_num += torch.sum(pred_label==target)
        label_num += target.numel()

    print('test accuracy:', correct_num/label_num)
    
    return correct_num/label_num

if __name__ == '__main__':
    test_acc = 0.
    net.load_state_dict(torch.load('./checkpoints/net_regu_27.pth'))
    for i in range(1):
        net.eval()
        test_acc = test()
    
    print('test accuracy:', test_acc)
