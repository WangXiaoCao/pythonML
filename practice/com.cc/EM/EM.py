# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GMM
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# # 生成来自两个高斯分布的数据
if __name__ == '__main__':
    # 第一个分布的均值
    mu1 = (0, 0, 0)
    # 创建一个维度为3的单位矩阵，作为方差
    cov1 = np.identity(3)
    # 根据均值与方差随机生成多元的正太分布的数据
    data1 = np.random.multivariate_normal(mu1, cov1, 100)
    # 第二个分布的均值
    mu2 = (2, 2, 1)
    # 设置方差
    cov2 = np.identity(3)
    # 根据均值与方差随机生成第二个多元的正太分布的数据
    data2= np.random.multivariate_normal(mu2, cov2, 100)
    # 将两组数据union
    data = np.vstack((data1, data2))

    # 设置迭代次数
    num_iter = 100
    # 样本数据的大小，n是样本数的个数，d是维度，这里是3维
    n, d = data.shape
    # 初始化参数，随机设定即可
    mu1 = np.random.standard_normal(d)  # 从三维的标准正态分布中随机取一个点作为初始均值
    mu2 = np.random.standard_normal(d)
    sigma1 = np.identity(d)  # 建立d维的单位矩阵作为初始化的方差
    sigma2 = np.identity(d)
    pi = 0.5 # 再拍脑门决定一下隐参数的值

    # # 进行EM算法
    for i in range(num_iter):
        # E-step
        # 根据均值方差创建两个正太分布
        norm1 = multivariate_normal(mu1, cov1)
        norm2 = multivariate_normal(mu2, cov2)
        # 分布计算每个样本点由两个分布产生的概率
        tau1 = pi*norm1.pdf(data)
        tau2 = (1 - pi)*norm2.pdf(data)
        # 计算由第一个分布产生的概率
        gamma = tau1/(tau1 + tau2)

        # M-step
        mu1 = np.dot(gamma, data)/sum(gamma)
        mu2 = np.dot((1-gamma), data)/sum(1-gamma)
        sigma1 = np.dot(gamma * (data - mu1).T, (data - mu1))/sum(gamma)
        sigma2 = np.dot((1-gamma) * (data - mu2).T, (data - mu2))/sum(1-gamma)
        pi = sum(gamma)/n
        # if i % 2 == 0:
        #     print i, ":\t",mu1, mu2

    print '类别概率：\t', pi
    print '均值：\t', mu1, mu2
    print '方差：\t', sigma1, sigma2

    g = GMM(n_components=2, covariance_type="full", n_iter=100)
    g.fit(data)
    print '类别概率:\t', g.weights_[0]
    print '均值:\n', g.means_, '\n'
    print '方差:\n', g.covars_, '\n'

    # 预测分类
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    # figsize设置画布的大小,facecolor设置画布背景色为白色
    fig = plt.figure(figsize=(14, 7), facecolor='w')
    # 121表示画1行2列中的第1个图：原始数据的分布图
    ax = fig.add_subplot(121, projection='3d')
    # x,y,z坐标是原始数据的第1,2,3列数据，c是颜色为蓝色，s是三点的大小，marker是三点的形状为圆，depthshade为颜色需不需要分深浅度
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    # 设置坐标的标记
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 设置图像的名称与此题大小
    ax.set_title(u'origin data', fontsize=18)

    # 画1行2列中的第2个图：预测数据的分类图
    ax = fig.add_subplot(122, projection='3d')
    # 提取出概率在分布1中比分布2中大的样本点
    c1 = tau1 > tau2
    # 将这点画在第2张图中，颜色为红色，形状是小圆
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    # 同理，拿出在分布2中概率大的样本点
    c2 = tau1 < tau2
    # 用绿色的小三角表示
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)

    # 自动调整图像大小，使两个图形比较紧凑
    plt.tight_layout()
    # 画出来吧，少年
    plt.show()

