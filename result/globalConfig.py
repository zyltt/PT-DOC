# coding=utf-8

import numpy as np
from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
class GlobalConfig:
    N = 0  # 现有种群规模
    L = 3  # 层数
    operatorNum = np.array([6,3,1])  # 每层计算图的个数
    c0 = 3  # 初始通道数
    flatOperatorMaps = []  # 还原出的所有扁平计算图
    resultNum = 3  # 需要搜索出的重复结构数目
    resGenetype = []  # 搜索出的genentype集合
    pointNum = [3,3,3]  # 每层的节点数
    maxMutateTime = 10000 # 最大突变次数
    P = Population()  # 初始种群
    Q = GenetypeQueue()  # controller选出的队列
    initMutateTime = 200 #初始化操作中执行突变的最大次数
    final_module = []#扁平图中节点之间的所有算子。
    channels = []#各节点的通道数。
    error_cal_mode = "mean"#误差计算方式，有max和mean两种。
    activation = "relu"#每个卷积后面的激活函数，有relu,sigmoid,tanh三种。
    corpus_size = 1000#原始数组语料库的大小
    max_diff_cal_time = 10#误差计算的迭代轮次
    h = 224 #featureMap的高度
    w = 224 #featureMap的宽度
    batch = 1#批次
    writer = None #csv书写器