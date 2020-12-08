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
from utils.certify import *
import torchvision.datasets as dset

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

net = InterConv(layer_num=1, hidden_num=50, class_num=2)

param_amount = 0
for p in net.named_parameters():
    if not 'resnet18' in p[0]:
        print(p[0])
        param_amount += p[1].numel()
print('total param amount:', param_amount)

net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

def generate_regularizer(in_list, out_list):
    length = len(in_list)
    reg_loss = 0.
    for i in range(length):
        xx = in_list[i]
        yy = out_list[i]
        for j in range(yy.shape[1]):
            grad_input = torch.autograd.grad(torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True)[0]
            
            coeff_mat = torch.ones_like(grad_input).cuda()
            coeff_mat[:, j] = coeff_mat[:, j] * (-1.)
            grad_input = grad_input*coeff_mat
            grad_input += .1
            grad_input[grad_input < 0.] = 0.
            reg_loss += torch.mean(grad_input**2)
         
    return reg_loss

def train(epoch):
    ### training
    loss_avg = 0.
    grad_loss_avg = 0.
    batch_idx = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        x = x.cuda()
        target = target.cuda()
        target[target==2] = 0
        target[target==7] = 1

        batch_idx += 1
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, target)
      
        in_list, out_list = net.reg_forward(num=512)
        reg_loss = generate_regularizer(in_list, out_list) 

        loss_total = loss + 1e2* reg_loss
        
        grad_loss_avg += reg_loss.detach().cpu().numpy()   
        loss_avg += loss.detach().cpu().numpy()
        loss_total.backward()
        optimizer.step()
            
    print('iter:', epoch, 'loss:', loss_avg/batch_idx, 'grad:', grad_loss_avg/batch_idx)

def test():            
    correct_num = 0.
    label_num = 0.
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_loader):
            x = x.cuda()
            target = target.cuda()
            target[target==2] = 0
            target[target==7] = 1
            
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            correct_num += torch.sum(pred_label==target)
            label_num += target.numel()

    print('test accuracy:', correct_num/label_num)
    
    return correct_num/label_num

if __name__ == '__main__':
    val_acc = 0.
    test_acc = 0.
    for i in range(20):
        net.train()
        train(i)

        net.eval()
        test_acc = test()
    
    #### Certify Monotonicity
    print(net.mono_submods_out[0].weight.data.shape)
    layer2_1 = nn.Linear(50, 1, bias=True)
    layer2_1.weight.data = net.mono_submods_out[0].weight.data[0, :].view(1, -1)
    mono_flag_1 = certify_grad_with_gurobi(net.mono_submods_in[0], layer2_1, 2, direction=[1,-1])

    layer2_2 = nn.Linear(50, 1, bias=True)
    layer2_2.weight.data = net.mono_submods_out[0].weight.data[1, :].view(1, -1)
    mono_flag_2 = certify_grad_with_gurobi(net.mono_submods_in[0], layer2_2, 2, direction=[-1,1])

    if mono_flag_1 and mono_flag_2:
        print('Certified Monotonic')
        torch.save(net.state_dict(), './checkpoints/net_regu_27.pth')
        print('test accuracy:', test_acc)
    else:
        print('Not Monotonic')
