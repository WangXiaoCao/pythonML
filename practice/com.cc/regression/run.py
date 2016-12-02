#!/usr/bin/python
# -*- coding:utf-8 -*-


from numpy import *
import matplotlib.pyplot as plt
import regression


# 读取与解析数据
f = open('D:\\Documents\\cc\\pythonL\\test.txt').readline().split('\t')
for line in f:
    print line


# xArr, yArr = regression.loadDataSet('D:\\Documents\\cc\\pythonL\\test.txt')
# print xArr

# # 调用方法计算最优参数值
# ws = regression.standRegres(xArr, yArr)
#
# # 计算预测值
# xmat = mat(xArr)
# ymat = mat(yArr)
# yhat = xmat * ws
#
# #绘制数据散点图
# fig = plt.figure()
# ax = fig.add_subplot(111)