#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np


data = []
label = []
with open("D:\Documents\cc\python\data\decisiontree.txt") as i_file:
    for line in i_file:
        tokens = line.strip().split(' ')
        data.append(float(tk) for tk in tokens[:-1])
        label.append(tokens[-1])
x = np.array(data)
label = np.array(label)
y = np.zeros(label.shape)

print label