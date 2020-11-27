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
import compas_loader as cp
import pandas as pd
import torch.utils.data as Data
from utils.networks import *
from utils.certify import *

X_train, y_train, X_test, y_test, start_index, cat_length = cp.load_data(get_categorical_info=True)
n = X_train.shape[0]
n = int(0.8*n)
X_val = X_train[n:, :]
y_val = y_train[n:]
X_train = X_train[:n, :]
y_train = y_train[:n]

feature_num = X_train.shape[1]
mono_feature = 4
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
X_val = torch.tensor(X_val).float()
y_train = torch.tensor(y_train).float().unsqueeze(1)
y_test = torch.tensor(y_test).float().unsqueeze(1)
y_val = torch.tensor(y_val).float().unsqueeze(1)
data_train = Data.TensorDataset(X_train, y_train)

data_train_loader = Data.DataLoader(
    dataset=data_train,      
    batch_size=256,     
    shuffle=True,               
    num_workers=2,              #
)

criterion = nn.BCEWithLogitsLoss()
net = MLP_relu(mono_feature=mono_feature, non_mono_feature=feature_num-mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num = 100, non_mono_hidden_num=100)
param_amount = 0
for p in net.named_parameters():
    print(p[0], p[1].numel())
    param_amount += p[1].numel()
print('total param amount:', param_amount)

net = net.cuda()
X_test = X_test.cuda()
y_test = y_test.cuda()
X_val = X_val.cuda()
y_val = y_val.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6, last_epoch=-1)

def generate_regularizer(in_list, out_list):
    length = len(in_list)
    reg_loss = 0.
    min_derivative = 0.0
    for i in range(length):
        xx = in_list[i]
        yy = out_list[i]
        for j in range(yy.shape[1]):
            grad_input = torch.autograd.grad(torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True)[0]
            grad_input_neg = -grad_input
            grad_input_neg += .2
            grad_input_neg[grad_input_neg < 0.] = 0.
            #reg_loss += torch.mean(grad_input_neg**2)
            #reg_loss += torch.max(grad_input_neg**2)
            if min_derivative < torch.max(grad_input_neg**2):
                min_derivative = torch.max(grad_input_neg**2)
    reg_loss = min_derivative
    return reg_loss

def train(epoch):
    ### training
    loss_avg = 0.
    grad_loss_avg = 0.
    batch_idx = 0
    for X, y in iter(data_train_loader):
        batch_idx += 1
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        X.requires_grad = True
        out = net(X[:, :mono_feature], X[:, mono_feature:])
        loss = criterion(out, y)
        
        in_list, out_list = net.reg_forward(feature_num=feature_num, mono_num=mono_feature, num=1024)
        reg_loss = generate_regularizer(in_list, out_list)
        
        loss_total = loss + 1e4 * reg_loss
        loss_avg += loss.detach().cpu().numpy()
        grad_loss_avg += reg_loss
        
        loss_total.backward()
        optimizer.step()
    #scheduler.step()    
    print('iter:', epoch, 'loss:', loss_avg/batch_idx, 'grad: ', grad_loss_avg/batch_idx)

def test():            
    ### test:
    out = net(X_test[:, :mono_feature], X_test[:, mono_feature:])
    out[out>0.] = 1.
    out[out<0.] = 0.
    print('test accuracy:', torch.sum(out==y_test)/float(y_test.numel()))
    
    return torch.sum(out==y_test)/float(y_test.numel())

def val():            
    ### val:
    out = net(X_val[:, :mono_feature], X_val[:, mono_feature:])
    out[out>0.] = 1.
    out[out<0.] = 0.
    print('val accuracy:', torch.sum(out==y_val)/float(y_val.numel()))
    
    return torch.sum(out==y_val)/float(y_val.numel())

if __name__ == '__main__':
    val_acc = 0.
    test_acc = 0.
    for i in range(200):
        net.train()
        train(i)
        
        net.eval()
        val_acc_cur = val()
        test_acc_cur = test()
        if (val_acc < val_acc_cur) or ((val_acc==val_acc_cur) and (test_acc<test_acc_cur)):
            test_acc = test_acc_cur
            val_acc = val_acc_cur
            torch.save(net.state_dict(), './net_compas.pth')

        print('best', val_acc, test_acc)

    net.load_state_dict(torch.load('./net_compas.pth'))    
    net.eval()
    mono_flag = certify_neural_network(net, mono_feature_num=mono_feature)
    if mono_flag:
        print('Certified Monotonic')
    else:
        print('Not Monotonic')
