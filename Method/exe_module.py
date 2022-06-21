# coding=utf-8

import copy
import csv
import tensorflow as tf
# 开启紧急执行，为了防止报错。
from DataStruct import edge
tf.compat.v1.enable_eager_execution()
import numpy as np

import DataStruct
from DataStruct.globalConfig import GlobalConfig
from queue import Queue
import tensorflow as tf
import torch

batch=GlobalConfig.batch
c=GlobalConfig.c0
h=GlobalConfig.h
w=GlobalConfig.w
activation=GlobalConfig.activation
#n：每批个数。h高度。w宽度。c通道数。
#注tensorflow默认使用数据格式为nhwc,pytorch默认使用数据格式为nchw，x的格式为nchw。因此tensorflow的输入需要转换。
def input_withDiffType(x, dtype,environment):
    if environment=="tensorflow":
        tensor_NCHW=tf.convert_to_tensor(x, dtype=dtype)
        tensor_NHWC=tf.transpose(tensor_NCHW,[0,2,3,1])
        return tensor_NHWC
    if environment=="pytorch":
        return torch.Tensor(x).type(dtype=dtype)
def createCorpus(n):
    q=Queue()
    for i in range(n):
        #nchw
        x = np.random.randn(batch, c, h, w)
        q.put(x)
    return q

def torch_module_executor(x):
    #各节点的张量。
    global channels
    global final_module
    tensors=[]
    #判断某张量是否有初始值。
    tensors_isnull=[True]*len(channels)
    tensors.append(input_withDiffType(x,dtype=torch.float32,environment="pytorch"))
    tensors_isnull[0]=False
    for i in range(len(channels)-1):
        #随意赋一个同类型的初始值
        tensors.append(copy.deepcopy(tensors[0]))
    final_point=0
    for eachOperation in final_module:
        fromIndex=eachOperation.fromIndex
        final_point=eachOperation.toIndex
        input=tensors[fromIndex]
        toIndex=eachOperation.toIndex
        #本NAS规定所有操作出通道数与入通道数相同。
        channel=channels[fromIndex]
        operator=eachOperation.operator
        #indentity
        if operator == -1:
            # print("pytorch执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                tensors[toIndex] = tensors[fromIndex].clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                # concat用于数组之间的操作，因此需要先进行类型转换。
                temp = torch.cat((tensors[toIndex], input), 1)
                tensors[toIndex] = temp.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        # 1 × 1 convolution of C channels
        elif operator == 1:
            # print("pytorch执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                #这是1*1卷积的代码
                # filter参数顺序:OutChannel、InChannel、H、W
                filter = torch.Tensor(np.ones([GlobalConfig.c0,channels[fromIndex],1,1])*(0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = thisresult.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                #这是1*1卷积的代码。
                filter = torch.Tensor(np.ones([GlobalConfig.c0, channels[fromIndex], 1, 1]) * (0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=0)
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        # 3 × 3 depthwise convolution
        elif operator == 2:
            # print("pytorch执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                # filter参数顺序:OutChannel、InChannel/groups、H、W
                filter = torch.ones((channels[fromIndex], 1, 3, 3),dtype=torch.float32) * 0.5
                #注：pytorch中dw卷积加pw卷积是普通卷积操作加groups来调节的。
                # thisresult = torch.nn.functional.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=channels[fromIndex])
                pad = torch.nn.ZeroPad2d(padding=(1,1,1,1))
                input = pad(input)
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                        groups=channels[fromIndex])
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(channels[fromIndex])(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = thisresult.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                filter = torch.ones((channels[fromIndex], 1, 3, 3),dtype=torch.float32) * 0.5
                # thisresult = torch.nn.functional.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=channels[fromIndex])
                pad = torch.nn.ZeroPad2d(padding=(1,1,1,1))
                input = pad(input)
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=1, padding=0,
                                                        groups=channels[fromIndex])
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(channels[fromIndex])(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        elif operator == 3:
            # print("pytorch执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                # filter参数顺序:OutChannel、InChannel/groups、H、W
                depthwise_filter = torch.ones((channels[fromIndex], 1, 3, 3),dtype=torch.float32) * 0.5
                pointwise_filter = torch.ones((GlobalConfig.c0, channels[fromIndex], 1, 1),dtype=torch.float32) * 0.5
                # depthwise_temp=torch.nn.functional.conv2d(input=input,weight=depthwise_filter,stride=1,padding=[1,1],groups=channels[fromIndex]).clone().detach()
                # pointwise_temp=torch.nn.functional.conv2d(input=depthwise_temp,weight=pointwise_filter,stride=1,padding=0).clone().detach()
                pad = torch.nn.ZeroPad2d(padding=(1,1,1,1))
                input = pad(input)
                depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                            padding=0, groups=channels[fromIndex]).clone().detach()
                pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                            padding=0).clone().detach()
                #归一化和relu
                pointwise_temp = torch.nn.BatchNorm2d(GlobalConfig.c0)(pointwise_temp)
                if activation=="relu":
                    pointwise_temp = torch.nn.functional.relu(pointwise_temp)
                elif activation=="sigmoid":
                    pointwise_temp = torch.sigmoid(pointwise_temp)
                else:
                    pointwise_temp = torch.tanh(pointwise_temp)
                tensors[toIndex]=pointwise_temp.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                depthwise_filter = torch.ones((channels[fromIndex], 1, 3, 3),dtype=torch.float32) * 0.5
                pointwise_filter = torch.ones((GlobalConfig.c0, channels[fromIndex], 1, 1),dtype=torch.float32) * 0.5
                pad = torch.nn.ZeroPad2d(padding=(1,1,1,1))
                input = pad(input)
                depthwise_temp = torch.nn.functional.conv2d(input=input, weight=depthwise_filter, stride=1,
                                                            padding=0, groups=channels[fromIndex]).clone().detach()
                pointwise_temp = torch.nn.functional.conv2d(input=depthwise_temp, weight=pointwise_filter, stride=1,
                                                            padding=0).clone().detach()
                #归一化和relu
                pointwise_temp = torch.nn.BatchNorm2d(GlobalConfig.c0)(pointwise_temp)
                if activation=="relu":
                    pointwise_temp = torch.nn.functional.relu(pointwise_temp)
                elif activation=="sigmoid":
                    pointwise_temp = torch.sigmoid(pointwise_temp)
                else:
                    pointwise_temp = torch.tanh(pointwise_temp)
                tensors[toIndex]=torch.cat((tensors[toIndex],pointwise_temp),1).clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        elif operator == 4:
            # print("pytorch执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=False)(input)
                tensors[toIndex] = result.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                result = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True)(input)
                temp = torch.cat((tensors[toIndex], result), 1)
                tensors[toIndex] = temp.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        elif operator == 5:
            # print("pytorch执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                #注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True,count_include_pad=False)(input)
                tensors[toIndex] = result.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                result = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True,count_include_pad=False)(input)
                temp = torch.cat((tensors[toIndex], result), 1)
                tensors[toIndex] = temp.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        # 3 × 3 convolution of C channels
        elif operator == 6:
            # print("pytorch执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                #这是3*3卷积的代码
                # filter参数顺序:OutChannel、InChannel、H、W
                filter = torch.Tensor(np.ones([GlobalConfig.c0,channels[fromIndex],3,3])*(0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                #归一化和relu

                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = thisresult.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                #这是3*3卷积的代码。
                filter = torch.Tensor(np.ones([GlobalConfig.c0, channels[fromIndex], 3, 3]) * (0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=1)
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
        # 5 × 5 convolution of C channels
        elif operator == 7:
            # print("pytorch执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                #这是1*1卷积的代码
                # filter参数顺序:OutChannel、InChannel、H、W
                filter = torch.Tensor(np.ones([GlobalConfig.c0,channels[fromIndex],5,5])*(0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=2)
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = thisresult.clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
            else:
                #这是5*5卷积的代码。
                filter = torch.Tensor(np.ones([GlobalConfig.c0, channels[fromIndex], 5, 5]) * (0.5))
                thisresult = torch.nn.functional.conv2d(input=input, weight=filter, stride=[1, 1], padding=2)
                #归一化和relu
                thisresult = torch.nn.BatchNorm2d(GlobalConfig.c0)(thisresult)
                if activation=="relu":
                    thisresult = torch.nn.functional.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = torch.sigmoid(thisresult)
                else:
                    thisresult = torch.tanh(thisresult)
                tensors[toIndex] = torch.cat((tensors[toIndex], thisresult), 1).clone().detach()
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tensors[toIndex])
    return tensors[final_point].clone().detach()


def tf_module_executor(x):
    #各节点的张量
    global channels
    global final_module
    tensors=[]
    #判断某张量是否有初始值。
    tensors_isnull=[True]*len(channels)
    tensors.append(input_withDiffType(x,dtype=tf.float32,environment="tensorflow"))
    tensors_isnull[0]=False
    for i in range(len(channels)-1):
        #随便赋一个同类型的初始值。
        tensors.append(copy.deepcopy(tensors[0]))

    final_point = 0
    for eachOperation in final_module:
        final_point = eachOperation.toIndex
        fromIndex=eachOperation.fromIndex
        input=tensors[fromIndex]
        toIndex=eachOperation.toIndex
        #in_channel表示操作的入通道数
        in_channel=channels[fromIndex]
        operator=eachOperation.operator
        #indentity
        if operator==-1:
            # print("tensorflow执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                tensors[toIndex]=copy.deepcopy(tensors[fromIndex])
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                #concat用于数组之间的操作，因此需要先进行类型转换。
                temp=tf.concat([tensors[toIndex],input],3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        #1 × 1 convolution of C channels
        elif operator==1:
            # print("tensorflow执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                #filter参数顺序:H、W、InChannel、OutChannel
                filter = tf.ones((1,1,channels[fromIndex],GlobalConfig.c0),dtype=tf.float32)*0.5
                thisresult =tf.nn.conv2d(input=input,filters=filter,strides=[1,1,1,1],padding='SAME')
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=True)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                tensors[toIndex]=copy.deepcopy(thisresult)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                filter = tf.ones((1, 1, channels[fromIndex], GlobalConfig.c0),dtype=tf.float32) * 0.5
                thisresult =tf.nn.conv2d(input=input,filters=filter,strides=[1,1,1,1],padding='SAME')
                # 激活函数和归一化
                thismean, thisvariance = tf.nn.moments(thisresult,axes=[0,1,2], keepdims=True)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                temp=tf.concat([tensors[toIndex],thisresult],3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        #3 × 3 depthwise convolution
        elif operator==2:
            # print("tensorflow执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                #filter参数顺序:H、W、InChannel、OutChannel的倍数
                filter = tf.ones((3,3,channels[fromIndex],1),dtype=tf.float32)*0.5
                thisresult = tf.nn.depthwise_conv2d(input=input,filter=filter,strides=[1,1,1,1],padding='SAME')
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                tensors[toIndex]=copy.deepcopy(thisresult)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                filter = tf.ones((3, 3, channels[fromIndex], 1),dtype=tf.float32) * 0.5
                thisresult = tf.nn.depthwise_conv2d(input=input, filter=filter, strides=[1,1,1,1], padding='SAME')

                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                temp = tf.concat([tensors[toIndex], thisresult], 3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            #为了debug,先把这两步注释掉。
            # tensors[toIndex] = tf.layers.batch_normalization(tensors[toIndex])
            # tensors[toIndex] = tf.nn.relu(tensors[toIndex])
        elif operator==3:
            # print("tensorflow执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                #各filter参数顺序:H、W、InChannel、OutChannel
                depthwise_filter = tf.ones((3, 3, channels[fromIndex], 1),tf.float32) * 0.5
                pointwise_filter = tf.constant(value = 0.5, shape = [1, 1, channels[fromIndex], GlobalConfig.c0],
                                               dtype = tf.float32)
                tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                    pointwise_filter=pointwise_filter,
                                                    strides=[1, 1, 1, 1], padding='SAME')

                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(tempresult,axes=[0,1,2],keepdims=False)
                tempresult=tf.nn.batch_normalization(tempresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    tempresult=tf.nn.relu(tempresult)
                elif activation=="sigmoid":
                    tempresult = tf.nn.sigmoid(tempresult)
                else:
                    tempresult = tf.nn.tanh(tempresult)

                tensors[toIndex]=copy.deepcopy(tempresult)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                depthwise_filter = tf.ones((3, 3, channels[fromIndex], 1),dtype=tf.float32) * 0.5
                pointwise_filter = tf.ones((1, 1, channels[fromIndex],GlobalConfig.c0),dtype=tf.float32)*0.5
                tempresult = tf.nn.separable_conv2d(input=input, depthwise_filter=depthwise_filter,
                                                    pointwise_filter=pointwise_filter,
                                                    strides=[1, 1, 1, 1], padding='SAME')
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(tempresult,axes=[0,1,2],keepdims=False)
                tempresult=tf.nn.batch_normalization(tempresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    tempresult=tf.nn.relu(tempresult)
                elif activation=="sigmoid":
                    tempresult = tf.nn.sigmoid(tempresult)
                else:
                    tempresult = tf.nn.tanh(tempresult)
                temp=tf.concat([tensors[toIndex],tempresult],3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        elif operator==4:
            # print("tensorflow执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                result=tf.nn.max_pool2d(input=input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
                tensors[toIndex]=copy.deepcopy(result)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                #concat用于数组之间的操作，因此需要先进行类型转换。
                result=tf.nn.max_pool2d(input=input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
                temp=tf.concat([tensors[toIndex],result],3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        elif operator==5:
            # print("tensorflow执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex]==True:
                tensors_isnull[toIndex]=False
                result=tf.nn.avg_pool2d(input=input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
                tensors[toIndex]=copy.deepcopy(result)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                #concat用于数组之间的操作，因此需要先进行类型转换。
                result=tf.nn.avg_pool2d(input=input,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME')
                temp=tf.concat([tensors[toIndex],result],3)
                tensors[toIndex]=copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        # 3 × 3 convolution of C channels
        elif operator == 6:
            # print("tensorflow执行了操作6", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                # filter参数顺序:H、W、InChannel、OutChannel
                filter = tf.ones((3, 3, channels[fromIndex], GlobalConfig.c0), dtype=tf.float32) * 0.5
                thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                tensors[toIndex] = copy.deepcopy(thisresult)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                filter = tf.ones((3, 3, channels[fromIndex], GlobalConfig.c0), dtype=tf.float32) * 0.5
                thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                # thisresult = tf.layers.Conv2D(filters=GlobalConfig.c0,kernel_size=3,strides=(1,1),padding='same',kernel_initializer=keras.initializers.Constant(value=0.5))(input)
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                temp = tf.concat([tensors[toIndex], thisresult], 3)
                tensors[toIndex] = copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
        elif operator == 7:
            # print("tensorflow执行了操作7", eachOperation.fromIndex, " ", eachOperation.toIndex)
            if tensors_isnull[toIndex] == True:
                tensors_isnull[toIndex] = False
                # filter参数顺序:H、W、InChannel、OutChannel
                filter = tf.ones((5, 5, channels[fromIndex], GlobalConfig.c0), dtype=tf.float32) * 0.5
                thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)
                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                tensors[toIndex] = copy.deepcopy(thisresult)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
            else:
                filter = tf.ones((5, 5, channels[fromIndex], GlobalConfig.c0), dtype=tf.float32) * 0.5
                thisresult = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
                # thisresult = tf.layers.Conv2D(filters=GlobalConfig.c0,kernel_size=3,strides=(1,1),padding='same',kernel_initializer=keras.initializers.Constant(value=0.5))(input)
                # 激活函数和归一化
                thismean,thisvariance=tf.nn.moments(thisresult,axes=[0,1,2],keepdims=False)
                thisresult=tf.nn.batch_normalization(thisresult,mean=thismean,variance=thisvariance,offset=None,scale=None,variance_epsilon=1e-5)

                if activation=="relu":
                    thisresult=tf.nn.relu(thisresult)
                elif activation=="sigmoid":
                    thisresult = tf.nn.sigmoid(thisresult)
                else:
                    thisresult = tf.nn.tanh(thisresult)
                temp = tf.concat([tensors[toIndex], thisresult], 3)
                tensors[toIndex] = copy.deepcopy(temp)
                # #log
                # print("tensor"+str(toIndex)+":")
                # print(tf.transpose(tensors[toIndex],[0,3,1,2]))
    return copy.deepcopy(tensors[final_point])



def module_executor():
    #先把需要的变量准备好
    global channels
    global final_module
    channels=copy.deepcopy(GlobalConfig.channels)
    final_module=copy.deepcopy(GlobalConfig.final_module)

    corpus=createCorpus(GlobalConfig.corpus_size)
    diff1_max=0
    diff2_max=0
    j=0
    while not corpus.empty() and j<GlobalConfig.max_diff_cal_time:
        x=corpus.get()
        x_temp_tf=copy.deepcopy(x)
        out_32_tf_NHWC=tf_module_executor(copy.deepcopy(x_temp_tf))
        out_32_tf_NCHW=tf.transpose(out_32_tf_NHWC,[0,3,1,2]).numpy().astype(np.float)
        torch_result = copy.deepcopy(x)
        torch_result=torch_module_executor(copy.deepcopy(torch_result)).numpy().astype(np.float)
        diff1 = np.mean(np.abs(out_32_tf_NCHW - torch_result))
        diff2 = np.max(np.abs(out_32_tf_NCHW - torch_result))
        if diff1>diff1_max:
            diff1_max=diff1
        if diff2>diff2_max:
            diff2_max=diff2
        j+=1
    return diff1_max,diff2_max
#用于记录单个算子的误差结果
#误差来源：tanh、sigmoid、3*3和5*5的普通卷积、所有的batchnorm
error_single_excutor_out=open(file='../error_single_excutor',mode='w',newline='')
error_single_excutor_writer=csv.writer(error_single_excutor_out)
activation_mode=["relu","sigmoid","tanh"]
for activation_mode_index in range(len(activation_mode)):
    GlobalConfig.activation=activation_mode[activation_mode_index]
    activation=GlobalConfig.activation
    error_single_excutor_writer.writerow(["当前激活函数为:"+activation_mode[activation_mode_index]])
    error_single_excutor_writer.writerow(["操作号","mean模式最大误差","max模式最大误差"])
    for i in range(-1,8,1):
        if i==0:
            continue
        GlobalConfig.channels=[3,3]
        GlobalConfig.final_module=[]
        GlobalConfig.final_module.append(DataStruct.edge.edge(0,1,i))
        temp1,temp2=module_executor()
        error_single_excutor_writer.writerow([i,temp1,temp2])