# coding=utf-8

from DataStruct.globalConfig import GlobalConfig
from .flatMap import toFlatMap
from .exe_module import module_executor
def calFitness(g):
    #todo
    toFlatMap(g)
    max_diff,mean_diff=module_executor()
    return max_diff,mean_diff
