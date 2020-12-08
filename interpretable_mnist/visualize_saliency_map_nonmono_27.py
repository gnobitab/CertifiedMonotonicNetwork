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
import matplotlib.pyplot as plt
import cv2
from matplotlib import colors

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root='/home/xcliu/datasets', train=True, transform=trans)
idx = (train_set.targets==2) | (train_set.targets==7)
train_set.targets = train_set.targets[idx]
train_set.data = train_set.data[idx]

test_set = dset.MNIST(root='/home/xcliu/datasets', train=False, transform=trans)
idx = (test_set.targets==2) | (test_set.targets==7)
test_set.targets = test_set.targets[idx]
test_set.data = test_set.data[idx]

batch_size = 128

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False)

### Full-batch-training
net = InterConv(layer_num=1, hidden_num=50, class_num=2)
net = net.cuda()
net.load_state_dict(torch.load('./checkpoints/net_noregu_27.pth'))

def normalize(mat):
    return (mat - np.min(mat))/(np.max(mat) - np.min(mat))

### Save Gradient Image Figure
def get_gradient(out, index, input_image):
    grad_map = torch.autograd.grad(torch.sum(out[:, index]), input_image, create_graph=True, allow_unused=True)[0]
    
    return grad_map
    
def get_test_saliency_avg():
    for batch_idx, (x, target) in enumerate(test_loader):
        x = x.cuda()
        target = target.cuda()
        target[target==2] = 0
        target[target==7] = 1
        if target[0] == 1:
            continue

        grad_map_collection = None
        for i in range(100): 
            noise = 0.01*torch.randn_like(x).cuda()
            noisy_x = x + noise
            noisy_x.requires_grad = True
            out, out_list = net.inter_forward(noisy_x)
            grad_map_cur = get_gradient(out_list[0], index=0, input_image=noisy_x)
            if torch.max(torch.abs(grad_map_cur)) == 0.0:
                continue
            else:
                if grad_map_collection is None:
                    grad_map_collection = grad_map_cur.clone()
                else:
                    grad_map_collection = torch.cat([grad_map_collection, grad_map_cur], dim=0)
        if grad_map_collection is None:
            continue 
        if grad_map_collection.shape[0] == 1:
            continue
        
        grad_map_collection = torch.mean(grad_map_collection, dim=0)
        
        grad_map = grad_map_collection.squeeze().detach().cpu().numpy()
        
        if np.max(np.abs(grad_map)) == 0.0:
            continue
        
        img = x.squeeze().detach().cpu().numpy()
        img = normalize(img)
        
        plt.axis('off')
        plt.imshow(img, cmap='gray', alpha=1.)
        plt.savefig('./figs/nonmono/%d_img.png'%batch_idx)
        plt.cla()
        
        grad_map_pos = grad_map.copy()
        grad_map_pos[grad_map_pos<0.] = 0.
        grad_map_neg = grad_map.copy()
        grad_map_neg[grad_map_neg>0.] = 0.
        grad_map_neg = -grad_map_neg
        
        plt.axis('off')
        plt.imshow(normalize(grad_map_pos), cmap='OrRd', alpha=1.)
        plt.savefig('./figs/nonmono/%d_pos_saliency.png'%batch_idx)
        plt.cla()
        
        plt.axis('off')
        plt.imshow(normalize(grad_map_neg), cmap='Blues', alpha=1.)
        plt.savefig('./figs/nonmono/%d_neg_saliency.png'%batch_idx)
        plt.cla()
        thres_pos = np.percentile((grad_map_pos*img).flatten(), 95) 
        img_modified = img.copy()
        img_modified[(grad_map_pos*img)>thres_pos] = 0.
        
        plt.axis('off')
        plt.imshow(img_modified, cmap='gray', alpha=1.)
        plt.savefig('./figs/nonmono/%d_img_modified.png'%batch_idx)
        plt.cla()

        print(batch_idx, target)
        if batch_idx > 20:
            break

if __name__ == '__main__':
   get_test_saliency_avg() 

