# coding=utf-8

from random import random
from DataStruct.globalConfig import GlobalConfig
from Method.calFitness import calFitness
class Worker:
    a=0
    #todo
    def __init__(self):
        return
    def excute(self):
        g=GlobalConfig.Q.pop()
        maxFitness,meanFitness=calFitness(g)
        if GlobalConfig.error_cal_mode=="max":
            g.fitness=maxFitness
            print("本轮误差为"+str(maxFitness))
        else:
            g.fitness=meanFitness
            print("本轮误差为"+str(meanFitness))
        GlobalConfig.P.append(g)
        return maxFitness,meanFitness