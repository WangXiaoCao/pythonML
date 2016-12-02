#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import operator


def createdataset():
    group = np.array ([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    dataSetSize = dataset.shape[0]  ##得到特征的维度
    diffMat = np.tile(inx, (dataSetSize, 1)) - dataset


def file2matrix(filename):
    fr =open(filename)
    arrayOfLines = fr.readlines()      ##获取文件
    numberOfLines = len(arrayOfLines)  ##获取文件行数
    returnMat = np.zeros((numberOfLines, 3))  ##创建一个行数*3的0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]  ## 追加入前面3列 -- 3个features
        classLabelVector.append(int(listFromLine[-1]))  ##加入最后一列--label
        index += 1
    return returnMat, classLabelVector
