import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import scipy.io as sio
import sys

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

'''
利用PCA对z进行分类，然后取center作为MUAPs
'''
USE_PCA = False
pca_dims = 3
# 对z按照列进行分类，相同类的叠加放在一起
nb_clusters = 7


MUAPs = np.load('./MUAPs.npy')
z = np.squeeze(np.load('./z.npy'))
A = np.squeeze(np.load('./A.npy'))

# 归一化
z_scaler = MinMaxScaler().fit(z)
z_scaled = z_scaler.transform(z)

# PCA降维
if USE_PCA:
    pca = PCA(n_components=pca_dims)
    z_scaled = pca.fit_transform(z_scaled)

kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(z_scaled)
z_T_labels = kmeans.labels_ 


#用字典的形式保存MUAPTs，例如MUAPs_clusters['0']里面全是为第0类的MUAPTs的波形，
MUAPs_clusters = dict()
for i in range(nb_clusters):
    MUAPs_clusters[str(i)] = []

for i, label in enumerate(z_T_labels):
    if np.all(np.abs(A[:, i])) > 0.8:
        MUAPs_clusters[str(label)].append(MUAPs[i])

for i in range(nb_clusters):
    MUAPs_clusters[str(i)] = np.array(MUAPs_clusters[str(i)]) 

# 保存MUAPs_clusters
np.save('./MUAPs_clusters.npy', MUAPs_clusters)


# 根据z_T_labels找到z中距离cluster center最近的z点的index，然后找到对应的MUAPTs
MUAPs_centers = dict()

for cluster in range(nb_clusters):
    z_cluster_index = np.arange(z.shape[0])[z_T_labels==cluster]
    z_cluster = z_scaled[z_cluster_index]
    z_cluster_distance = np.sum(kmeans.transform(z_cluster), axis=1)
    z_center_index = z_cluster_index[np.argsort(z_cluster_distance)[-1]]
    MUAPs_centers[str(cluster)] = np.expand_dims(MUAPs[z_center_index], 0)

np.save('./MUAPs_centers.npy', MUAPs_centers)








