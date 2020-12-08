import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as Data

def certify_grad_with_gurobi(first_layer, second_layer, mono_feature_num, direction=None):
    ### default: certify monotonically increasing; if set direction=[1, -1], 1 means inc, -1 means dec
    mono_flag = True
    w1 = first_layer.weight.data.detach().cpu().numpy().astype('float64')
    w2 = second_layer.weight.data.detach().cpu().numpy().astype('float64')
    b1 = first_layer.bias.data.detach().cpu().numpy().astype('float64')
    b2 = second_layer.bias.data.detach().cpu().numpy().astype('float64')
    feature_num = w1.shape[1]
    
    for p in range(mono_feature_num):
        if direction is not None:
            if direction[p] == -1:
                w2 = -w2

        fc_first = w1[:, p]
        
        m_up = np.sum(np.maximum(w1, 0.0), axis=1) + b1
        m_down = -np.sum(np.maximum(-w1, 0.0), axis=1) + b1
        h = np.concatenate((-b1, b1 - m_down), axis=0)
        
        G_z = np.zeros((w1.shape[0] * 2, w1.shape[0])) 
        G_x = np.zeros((w1.shape[0] * 2, feature_num))
        for i in range(w1.shape[0]):
            G_x[i, :] = w1[i, :]
            G_z[i, i] = -m_up[i]
        
        for i in range(w1.shape[0]):
            G_x[i + w1.shape[0], :] = -w1[i, :]
            G_z[i + w1.shape[0], i] = -m_down[i]
        
        end_1 = time.time()
        m = gp.Model("matrix1")
        m.Params.Threads = 48
        m.Params.OutputFlag = 0
        z = m.addMVar(shape=w1.shape[0], vtype=GRB.BINARY, name="z")
        a = m.addMVar(shape=w2.shape[0], lb=0.0, vtype=GRB.CONTINUOUS, name="a")
        x = m.addMVar(shape=feature_num, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        
        obj_mat = np.zeros((w2.shape[0], w1.shape[0]))
        for q in range(w2.shape[0]): 
            fc_last  = w2[q, :]
            c = fc_last * fc_first
            obj_mat[q, :] = c
        
        one_array = np.ones((w2.shape[0]))
        m.addConstr(one_array.T @ a == 1., name="constraint_a")
        m.addConstr((G_z @ z + G_x @ x) <= h, name="constraint")
        m.setObjective(a @ (obj_mat @ z), GRB.MINIMIZE)
        m.optimize()
        print('Obj: %g' % m.objVal, 'a:', np.array(a.x))
        if m.objVal < 0.:
            print('Non-monotonic')
            mono_flag = False
            break
        
        if direction is not None:
            if direction[p] == -1:
                w2 = -w2

    return mono_flag

def certify_neural_network(model, mono_feature_num):
    '''The function to certify monotonicity of a trained model'''
    layer_num = len(model.mono_submods_in)+1
    layer_in = []
    layer_out = []
    layer_mono_feature_num = []
    
    layer_in.append(model.mono_fc_in)
    layer_mono_feature_num.append(mono_feature_num)
    for i in range(layer_num-1):
        layer_in.append(model.mono_submods_in[i])
        layer_out.append(model.mono_submods_out[i])
        layer_mono_feature_num.append(10)
    layer_out.append(model.mono_fc_last)

    for i in range(layer_num):
        mono_flag = certify_grad_with_gurobi(layer_in[i], layer_out[i], layer_mono_feature_num[i])
        if not mono_flag:
            return False

    return True

