from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import random
import numpy as np
import operator

data = pd.read_table(r"Haberman's Survival Data.txt", sep=",", header=None, names=['opAge', 'opYear', 'cellNum', 'status'], engine="python")
data1 = data[data['status'] == 1] # status为1的数据
data2 = data[data['status'] == 2] # status为2的数据

fig = plt.figure(figsize=(16, 12))
ax = fig.gca(projection="3d") #get current axis
ax.scatter(data1['opAge'], data1['opYear'], data1['cellNum'], c='r', s=100, marker="*", label="survived 5 years or longer") #status为1的样本的散点图
ax.scatter(data2['opAge'], data2['opYear'], data2['cellNum'], c='b', s=100, marker="^", label="died within 5 year") #status为2的样本的散点图
ax.set_xlabel("operation age", size=15)
ax.set_ylabel("operation year", size=15)
ax.set_zlabel("cell number", size=15)
ax.set_title('Haberman\'s Survival Scatter Plot', size=15, weight='bold')
ax.set_zlim(0, 30)
ax.legend(loc="lower right", fontsize=15)
plt.show()
samples=data

# 归一化 (x-min)/(max-min)∈[0,1]
def autoNorm(dataSet):
    minVals = samples.min(axis=0) # 按列求最小值，即求每个属性的最小值
    maxVals = samples.max(axis=0) # 求每个属性的最大值
    factors = maxVals - minVals # 归一化因子
    sNum = dataSet.shape[0]  # 数据集的行数，即样本数
    normDataSet = (dataSet - np.tile(minVals, (sNum, 1))) / np.tile(factors, (sNum, 1))  # 先将minVals和归一化因子转换成与dataSet相同的shape，再做减法和除法运算，最终归一化后的数据都介于[0,1]
    return normDataSet

testIdxs = random.sample(range(0, len(samples), 1), 10)
#testIdxs = random.sample(range(0, len(samples), 1), len(samples) * 1 / 10)  # 随机选取testing data的索引# 随机选取testing data的索引
print(testIdxs)
testingSet = samples.ix[testIdxs]  # 根据索引从样本集中获取testing data
idxs = range(0, len(samples), 1)  # 总的数据索引序列
print(idxs)
#以下for循环是从总的数据索引序列中将testing data的索引去除
'''for i in range(10):
    print(testIdxs[i])
    idxs.remove(testIdxs[i])'''
print(list(set(idxs).difference(set(testIdxs))))
idxs=list(set(idxs).difference(set(testIdxs)))
trainData = samples.ix[idxs]  # 获取用作训练的数据集
print(trainData)
#inX: 目标样本
#dataSet: 用来找k nearest neighbor的数据集，labels是该数据集对应的类别标签，dataSet和labels的索引是一一对应的
def classifyKNN(inX, dataSet, labels, k):
    #以下代码是为了防止出现这种情况：dataSet和labels的索引不是从0开始的有序自然数，导致在argsort排序的时候出现错乱，因为argsort排序结果是从0开始的自然数，因此首先需要重置dataSet和labels的索引，使其索引变为依次从0开始自然数。
    nDataSet = np.zeros((dataSet.shape[0], dataSet.shape[1])) #与dataSet同型的0矩阵
    j = 0
    for i in dataSet.index:
        nDataSet[j] = dataSet.ix[i]
        j += 1
    nDataSet = pd.DataFrame(nDataSet)

    nLabels = np.zeros(labels.shape[0]) #与labels同型的0向量
    h = 0
    for i in labels.index:
        nLabels[h] = labels.ix[i]
        h += 1

    dataSetNum = nDataSet.shape[0]  # 样本数(DataFrame行数)
    diffMat = np.tile(inX, (dataSetNum, 1)) - nDataSet #目标样本与参照样本集的差，对应属性相减，结果为与nDataSet同型的矩阵
    sqDiffMat = diffMat ** 2  #平方
    sqDistances = sqDiffMat.sum(axis=1) #矩阵sqDiffMat的列之和，即目标样本与样本集中每个样本对应属性的差值的平方和
    distances = sqDistances ** 0.5 #平方根，欧氏距离，即目标样本与每个样本点的距离
    sortedDistanceIdx = distances.argsort()  # 距离从小到大的索引值，sortedDistanceIdx的索引是从0开始的自然数，sortedDistanceIdx的值表示对应的distance的索引，比如sortedDistanceIdx[0]是150，表示最小的距离在distances中的索引是150
    classCount = {}
    for i in range(k):
        #找出distance最小的k个索引，然后在nLabels中获取其对应类别
        voteLabel = nLabels[int(sortedDistanceIdx[i])]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    #classCount字典中存放了统计的label和对应的出现次数
    sortedClassCount = sorted(classCount.items (), key=operator.itemgetter(1), reverse=True) #倒序
    return sortedClassCount[0][0]  #出现次数最大的label
#返回在该验证集上的错误率
def train(trainingSet, validationSet, kn):
    errorCount = 0
    vIdxs = validationSet.index
    #遍历验证集，对每个样本使用KNN
    for i in range(0, len(validationSet)):
        pred = classifyKNN(validationSet.loc[vIdxs[i], ['opAge', 'opYear', 'cellNum']], trainingSet[['opAge', 'opYear', 'cellNum']], trainingSet['status'], kn)
        if (pred != validationSet.at[vIdxs[i], 'status']):
            errorCount += 1
    return errorCount / float(len(validationSet))
# dataSet：用来交叉验证的数据集，idxs是对应的索引序列
# k: k折交叉验证
# kn: kn近邻
def crossValidation(dataSet, idxs, k, kn):
    step = int(len(idxs) / k)
    
    errorRate = 0
    for i in range(k):
        validationIdx = []
        for i in range(i * step, (i + 1) * step):
            validationIdx.append(idxs[i])
        validationSet = dataSet.ix[validationIdx]  # 获得验证集数据
        temp = idxs[:]
        for i in validationIdx:  # 把验证集的索引去除
            temp.remove(i)
        trainingSet = dataSet.ix[temp]  # 获取训练集数据
        errorRate += train(trainingSet, validationSet, kn)
    aveErrorRate = errorRate / float(k)
    return aveErrorRate
def predict(trainingSet, testingSet, kn):
    errorCount = 0
    for i in range(0, len(testingSet)):
        vIdxs = testingSet.index
        pred = classifyKNN(testingSet.loc[vIdxs[i], ['opAge', 'opYear', 'cellNum']], trainingSet[['opAge', 'opYear', 'cellNum']], trainingSet['status'], kn)
        print ("The prediction label is %s"%(pred))
        print ("The real label is %s"%(testingSet.at[vIdxs[i], 'status']))
        if (pred != testingSet.at[vIdxs[i], 'status']):
            errorCount += 1
    return errorCount 
print ("The cross validation error ratio is %d" %crossValidation(trainData, idxs, 10, 3))
print ("The testing data error ratio is %d"%predict(samples,testingSet,3))
print(predict(samples,testingSet,3)/len(testingSet))
