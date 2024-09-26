from collections import defaultdict
import numpy as np
import torch

'''
def seperate(Z, y_pred, n_clusters):
    # 根据预测标签 y_pred，将矩阵 Z 中的行按聚类标签分离，并存储到一个字典 Z_seperate 中，其中字典的键是聚类标签，值是对应标签的行列表。
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j].cpu().detach().numpy())
                Z_new[j][:] = Z[j].cpu().detach().numpy()
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    
    # 该代码的目的是根据聚类标签将数据矩阵 Z 分离，然后对每个聚类的数据进行 SVD 分解，最终将所有聚类的分解结果整合到一个矩阵 D 中，用于初始化某种计算或算法。
    # num_expert * d = n_z

    Z_seperate = seperate(Z, y_pred, n_clusters)
    Z_full = None
    U = np.zeros([n_clusters * d, n_clusters * d])
    print("Initialize D")
    # 对每个聚类中的数据进行奇异值分解 (SVD)，并将结果中的左奇异矩阵 u 的前 d 列赋值给 U 矩阵中的相应位置。
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        U[:,i*d:(i+1)*d] = u[:,0:d]
    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D
'''


def seperate(Z, y_pred, n_clusters):
    # 根据预测标签 y_pred，将矩阵 Z 中的行按聚类标签分离，并存储到一个字典 Z_seperate 中，其中字典的键是聚类标签，值是对应标签的行列表。
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = {}
    for i in range(n_clusters):
        Z_seperate[i] = []
    for j in range(len(y_pred)):
        cluster_label = y_pred[j]
        Z_seperate[cluster_label].append(Z[j].cpu().detach().numpy())  # 将Tensor转换为NumPy数组并添加到列表中
    return Z_seperate

def Initialization_D(Z, y_pred, n_clusters, d):
    '''
    该代码的目的是根据聚类标签将数据矩阵 Z 分离，然后对每个聚类的数据进行 SVD 分解，最终将所有聚类的分解结果整合到一个矩阵 D 中，用于初始化某种计算或算法。
    num_expert * d = n_z
    '''
    Z_seperate = seperate(Z, y_pred, n_clusters)
    U = np.zeros([n_clusters * d, n_clusters * d])
    print("Initialize D")
    for i in range(n_clusters):
        # 对每个聚类中的数据进行 SVD 分解
        data = np.array(Z_seperate[i])  # 转换为NumPy数组
        u, ss, v = np.linalg.svd(data.T, full_matrices=False)
        U[:, i*d:(i+1)*d] = u[:, :d]  # 直接取前 d 列赋值给 U

    D = U
    print("Shape of D: ", D.T.shape)
    print("Initialization of D Finished")
    return D

