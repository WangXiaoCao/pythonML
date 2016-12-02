#!/usr/bin/python
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt

# 第一个参数是框的形状，fc是框的透明度
decisionNode = dict(boxstyle="sawtooth", fc='1')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    # 整片画布的背景颜色
    fig = plt.figure(1, facecolor = 'white')
    # 把把绘图区清空
    fig.clf()
    #第一个参数是图形占画布的大小，第二个是设置是否需要边框
    createPlot.ax1 = plt.subplot(222, frameon=True)
    # 第一个参数是注解的文字内容，两个坐标是箭头首尾的坐标，第三个参数是之前定义的文本框
    plotNode(U'decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(U'leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    # 画出来吧少年
    plt.show()