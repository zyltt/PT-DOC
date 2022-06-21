# coding=utf-8

from Method.asyncTournamentSelect import asyncTournamentSelect
from DataStruct.globalConfig import GlobalConfig
from Method.mutation import mutation
class Controller:
    #todo
    a=0
    def __init__(self):
        return
    def excute(self):
        #todo
        g=asyncTournamentSelect(GlobalConfig.P)
        mutation(g)
        GlobalConfig.Q.push(g)
        return