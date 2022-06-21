# coding=utf-8

from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
from DataStruct.globalConfig import GlobalConfig
from DataStruct.controller import Controller
from DataStruct.worker import Worker
from Method.flatMap import toFlatMap
from Method.initialize import initialize
from Method.exe_module import module_executor
from Method.util import getFinalModule_in_str,getChannels_in_str
import csv

def globalInit(error_cal_mode , activation):
    # step1:配置globalConfig
    print("开始计算"+error_cal_mode+"_"+activation)
    out = open(file='./' + 'result_' + error_cal_mode + '_' + activation +'.csv' , mode='w', newline='')
    writer = csv.writer(out,delimiter = ",")
    GlobalConfig.N = 0
    GlobalConfig.flatOperatorMaps = []
    GlobalConfig.resGenetype = []
    GlobalConfig.P = Population()
    GlobalConfig.Q = GenetypeQueue()
    GlobalConfig.final_module = []
    GlobalConfig.channels = []
    GlobalConfig.error_cal_mode = error_cal_mode
    GlobalConfig.activation = activation
    GlobalConfig.writer = writer
    writer.writerow(["轮次","max模式最大误差","mean模式最大误差","各点通道数","算子组合"])


modes = ["mean","max"]
activations = ["relu","sigmoid","tanh"]

for thisMode in modes:
    for thisActivation in activations:
        globalInit(thisMode,thisActivation)
        print("正在初始化种群")
        initialize(GlobalConfig.P)
        print("种群初始化完成")
        print("开始构建controller节点")
        controller = Controller()
        print("controller节点构建完成")
        print("开始构建worker节点")
        worker = Worker()
        print("worker节点构建完成")

        #主流程
        t = 0
        print("开始进行突变")
        while(t < GlobalConfig.maxMutateTime):
            controller.excute()
            maxFitness , meanFitness = worker.excute()
            print("第" + str(t) + "轮已经完成")
            GlobalConfig.writer.writerow([str(t),str(meanFitness),str(maxFitness),getChannels_in_str(),getFinalModule_in_str()])
            t = t + 1

        #最后的筛选
        while(len(GlobalConfig.resGenetype) < GlobalConfig.resultNum):
            controller.excute()
            thisg=GlobalConfig.Q.pop()
            GlobalConfig.resGenetype.append(thisg)