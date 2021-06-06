"""
# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 0001 14:37
# @Author  : 源来很巧
# @FileName: code1.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44793283
"""

import pandas as pd
data=pd.read_excel("zscoredata.xls")
data_mean=data.mean()
data_std=data.std()
# print("平均值：\n",data_mean)
# print("标准差：\n",data_std)
data_score=(data-data.mean())/data.std()
# print(data_score)
data_score.to_csv("data_score.csv")
'''标准化的另一种方法'''
# from sklearn.preprocessing import StandardScaler
# data = StandardScaler().fit_transform(data)
# # np.savez('../tmp/airline_scale.npz',data)
# print('标准化后LRFMC五个属性为：\n',data[:5,:])



'''聚类'''
import pandas as pd
from sklearn.cluster import KMeans  # 导入kmeans算法
# 读取标准化后的数据
airline_scale = data_score
k = 5  # 确定聚类中心数
# 构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters = k,random_state=10)
fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练
# 查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_  # 聚类中心
print('各类聚类中心为：\n',kmeans_cc)
kmeans_labels = kmeans_model.labels_  # 样本的类别标签
print('各样本的类别标签为：\n',kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同类别样本的数目
print('最终每个类别的数目为：\n',r1)
# 输出聚类分群的结果
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_,\
             columns = ['ZL','ZR','ZF','ZM','ZC'])   # 将聚类中心放在数据框中
cluster_center.index = pd.DataFrame(kmeans_model.labels_ ).\
                  drop_duplicates().iloc[:,0]  # 将样本类别作为数据框索引
print(cluster_center)


import numpy as np
import matplotlib.pyplot as plt
# 客户分群雷达图
labels = ['ZL','ZR','ZF','ZM','ZC']
legen = ['客户群' + str(i + 1) for i in cluster_center.index]  # 客户群命名，作为雷达图的图例
lstype = ['-','--',(0, (3, 5, 1, 5, 1, 5)),':','-.']
kinds = list(cluster_center.iloc[:, 0])
#print(len(kinds))
# 由于雷达图要保证数据闭合，因此再添加L列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])
print(cluster_center.iloc[:, 0:])
# 分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
print(angle)
angle = np.concatenate((angle, [angle[0]]))
labels = np.concatenate((labels, [labels[0]]))
# 绘图
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, polar=True)  # 以极坐标的形式绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 画线
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2,label=kinds[i])
    #print(centers[i],kinds[i])
# 添加属性标签
ax.set_thetagrids(angle*180/np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.savefig("雷达图.jpg")
plt.show()


