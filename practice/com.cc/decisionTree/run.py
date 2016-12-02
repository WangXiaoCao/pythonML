#!/usr/bin/python
# -*- coding:utf-8 -*-

import tree
import treePlotter


feature, labels = tree.create_data_set()
# en = tree.calcShannomEnt(feature)
# print en
# print feature
# print  labels

# feature[0][-1] = "maybe"
# en2 = tree.calcShannomEnt(feature)
# print feature
# print en2

# split = tree.splitDataSet(feature,0, 0)
# print tree.splitDataSet(feature,0, 0)
# print tree.splitDataSet(feature,0, 1)

# bestFeature = tree.chooseBestFeature(feature)
# print bestFeature

myTree = tree.create_tree(feature, labels)
print myTree

# treePlotter.createPlot()
feature, labels = tree.create_data_set()
pre = tree.classify(myTree, labels, [1,0])
print pre




