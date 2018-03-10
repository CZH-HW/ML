'''
Linear Regression Model

Creat on Jan 17, 2018

@author CZH
'''

import numpy as np 

def loadDataSet(fileName):
    """
    数据导入函数
    打开一个用tab键分开的txt文件
    """

    num = len(open(fileName).readline().split('\t'))-1      # 获取每行的数据个数 - 1
    data_Mat = []        # 存储x特征数据的列表
    label_Mat = []       # 存储y标签数据的列表

    content = open(fileName) 
    for line in content.readlines():
        # 读取文件中每一行的内容，并获取每行内容以tab键分隔的数据，将其分别append到dataMat和labelMat中
        curLine = line.strip().split('\t') 
        line_Arr = []                         # 用来临时存放x特征数据的列表
        for i in range(num):
            line_Arr.append(float(curLine[i]))
        data_Mat.append(line_Arr)
        label_Mat.append(float(curLine[-1]))
    return data_Mat, label_Mat               # 返回数组



def loadTxtData(fileName):
    """可直接使用 data = np.loadtxt(fileName)导入数据"""
    num = len(open(fileName).readline().split('\t'))-1      # 获取每行的数据个数 - 1
    data = np.loadtxt(fileName)             # 返回数据的数组
    data_Mat = data[:,:num]
    label_Mat = data[:,num]
    return data_Mat, label_Mat              # 返回数组


def standRegression(xArr, yArr):
    """标准回归函数"""
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('此矩阵奇异，不可逆')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k = 1.0):
    """Locally Weighted Linear Regression 局部加权线性回归"""
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(yMat)[0]                    # 获取对角矩阵的行列数
    weight_Mat = np.mat(np.eye(m))           # 创建对角矩阵
    for i in range(m):
        diffMat = testPoint - xMat[i,:]      # diffMat为样本点与预测点距离，以向量表示
        weight_Mat[i,i] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))    # 权重大小以指数级衰减
    xTwx = xMat.T * weight_Mat * xMat
    if np.linalg.det(xTwx) == 0.0:
        print('此矩阵奇异，不可逆')
        return
    ws = xTwx.I * (xMat.T * (weight_Mat * yMat))
    return (testPoint * ws)[0,0]             # 返回测试点的预测值


def lwlrTest(testArr, xArr, yArr, k = 1.0):
    """获取测试数据集中所有点的估计值"""
    m = np.shape(testArr)[0]                 # 获取测试数据集的测试样本数
    yHat = np.zeros(m)                       # yHat是用来存放预测值的数组
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    '''计算预测值与实际值误差的大小'''
    return ((yArr - yHatArr)**2).sum()

def ridgeRegression(xMat, yMat, lam=0.2):
    '''岭回归,求单个lambda对应的回归系数'''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeRegressionTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 特征和标签标准化处理
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)     # 求均值
    xVar = np.var(xMat, 0)       # 求方差
    xMat = (xMat - xMean)/xVar
    numTest = 30                 # 规定测试的不同lambda的数量
    wMat = np.zeros((numTest, np.shape(xMat)[1]))      # wMat用来存放不同lambda对应的回归系数的矩阵
    for i in range(numTest):
        ws = ridgeRegression(xMat, yMat, np.exp(i-10))    # lambda以指数级变化
        wMat[i,:] = ws.T
    return wMat
    






    
        






