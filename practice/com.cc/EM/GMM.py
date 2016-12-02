# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cross_validation import train_test_split
from sklearn.mixture import GMM
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d


if __name__ == '__main__':
    # 读入数据（性别，身高，体重）
    data = np.loadtxt('D:\Documents\cc\python\data\HeightWeight.csv',
                      dtype = np.float, delimiter=',', skiprows=1)
    # 将数据分成两部分y:性别标签列，x:身高与体重2列,[1, ]表示第一部分取[0,1)列，第二部分取剩下的列
    y, x = np.split(data, [1, ], axis=1)
    # 将数据分成训练集与测试集，random_state保证了随机池相同
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
    # 用生成一个GMM，由两个分布组成，tol是阀值
    gmm = GMM(n_components=2, covariance_type='full', tol=0.0001, n_iter=100, random_state=0)
    # 分别取出身高与体重的最大值与最小值
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    # 使身高与体重的训练数据去拟合刚刚创建的gmm
    gmm.fit(x)
    # 打印均值，方差看一看
    print '均值 = \n', gmm.means_
    print '方差 = \n', gmm.covars_
    # 对训练数据与测试数据都带入gmm做预测
    y_hat = gmm.predict(x)
    y_test_hat = gmm.predict(x_test)
    # 以为gmm做预测是不会分两个分布的先后顺序的，所以要保证是顺序颠倒的时候女性仍然表示为0
    change = (gmm.means_[0][0] > gmm.means_[1][0])
    if change:
        z = y_hat == 0
        y_hat[z] = 1
        y_hat[~z] = 0
        z = y_test_hat == 0
        y_test_hat[z] = 0
        y_test_hat[~z] = 0
    # 求训练与测试集的准确率
    acc = np.mean(y_hat.ravel() == y.ravel())
    acc_test = np.mean(y_test_hat.ravel() == y_test.ravel())
    acc_str = u'训练集准确率：%.2f%%' % (acc * 100)
    acc_test_str = u'测试集准确率：%.2f%%' % (acc_test * 100)
    print acc_str
    print acc_test_str

    # 创建浅色的颜色（用来做背景）和深色的颜色（用来表示点）
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    # 分别取两个特征的最大值与最小值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    # 在画布上画上横纵坐标网格各500
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    # 将两个特征的数铺平成一列
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    # 用gmm做预测y值
    grid_hat = gmm.predict(grid_test)
    # 将预测出来的y值调整大小成与x1一致
    grid_hat = grid_hat.reshape(x1.shape)
    if change:
        z = grid_hat == 0
        grid_hat[z] = 1
        grid_hat[~z] = 0
    # 画一个9*7的画布，背景色为白色
    plt.figure(figsize=(9, 7), facecolor='w')
    # 用之前设置的浅颜色来做两个分布的区分区域
    plt.pcolormesh(x1, x2, grid_hat,cmap=cm_light)
    # 画训练样本点，用圆表示，并使用之前设置的深色表示
    plt.scatter(x[:, 0], x[:, 1], s=50, c=y, marker='o', cmap=cm_dark, edgecolors='k')
    # 画训练样本点，用三角形表示，并使用之前设置的深色表示
    plt.scatter(x_test[:, 0], x_test[:, 1], s=60, c=y_test, marker='^', cmap=cm_dark, edgecolors='k')
    # 奔跑吧，少年
    plt.show()

