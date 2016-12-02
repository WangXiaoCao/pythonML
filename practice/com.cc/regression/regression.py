#!/usr/bin/python
# -*- coding:utf-8 -*-

from numpy import *

# # # 读取文件，并将数据转换成numpy能够解析的特征矩阵，和标签矩阵
def loadDataSet(fileName):
    # # 维度的个数
    numFeat = len(open(fileName).readline().split('\t')) - 2
    print numFeat
    # # 创建一个空的特征矩阵，一个空的标签矩阵
    dataMat = []; labelMat = []
    # # 根据文件名导入文件
    fr = open(fileName)
    # # 对文件中的每行都循环提取特征数据到特征矩阵，提取标签数据到标签矩阵
    for line in fr.readline():
        lineArr = []
        curLine = line.strip().split('\t')
        # # 将特征数据一个一个追加到一个数组
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        # # 将特征数组追加到特征矩阵中
        dataMat.append(lineArr)
        # # 将标签元素最佳到标签矩阵中
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# # # 根据最小二乘法计算最优参数
def standRegres(xArr, yArr):
    # # 将xy数组分别保存到矩阵中
    xMat = mat(xArr); yMat = mat(yArr).T
    # # 计算x乘以x的转置
    xTx = xMat.T*xMat
    # # 判断xTx的行列式是否为0 ，是的话就打印东东并返回
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    # # 若以上条件不成立，则根据最小二乘公式计算最优权重θ，并返回
    ws = xTx.I * (xMat * yMat)
    return ws
