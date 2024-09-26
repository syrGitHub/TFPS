import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_constraint1(torch.nn.Module):
    '''
    该约束确保矩阵 d 的列近似正交，且每列向量的长度接近1。即使得 d 近似于正交矩阵。
    '''
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).to(d.device)  # .cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):
    '''
    d.shape: [20, 2]
    dim: 10
    n_clusters: 2
    该约束确保不同聚类之间的矩阵 d 的子块尽量独立，即相互正交。其目的是减少聚类之间的相互干扰。
    '''
    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[1], d.shape[1]).to(d.device)  # S [2, 2]
        zero = torch.zeros(dim, dim)        # zero [10, 10]
        # print('S.shape, zero.shape:',  S.shape, zero.shape)
        # 将矩阵 S 按块分为 n_clusters 个大小为 dim×dim 的子矩阵，并将这些子矩阵设为 zero。
        for i in range(n_clusters):
            #print(i, dim, i*dim, (i+1)*dim)
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint


