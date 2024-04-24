
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 其中 1.0给偏置项b，第二项，第三项输入，
        labelMat.append(int(lineArr[2]))  # 数据第三项是标签即输出
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001 # 学习率
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) # 三个输入一个输出
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     # 100*3 *3*1 得到100*1
        error = (labelMat - h)              #vector subtraction error是100*1
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult 3*100
        print(weights)
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] #只需要知道有几个变量
    xcord1 = []; ycord1 = [] #用于获得在类别1下的特征1和特征2数据
    xcord2 = []; ycord2 = []#用于获得在类别0下的特征1和特征2数据
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #画上散点图
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 1)
    y = (-weights[0]-weights[1]*x)/weights[2] #这句话类似与w0 + w1x1 + w2x2 = class，得到一个区分线，其中x2表示y 这里只是用一种反解的方法去表示特征2
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels): #这个随机梯度下降在每次计算时候只是拿了其中一行数据去做权重的更新
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) # 这里的h是一个数值，
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i] #这里是数乘，而在gradAscent中是用了点积的方法
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m)) #python3产生的是range对象，需要列表化才能删除
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 学习率会随着迭代次数变小
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant #方法将随机生成一个实数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex]) #每次遍历数据都会删除随机梯度下降所用到的那个数据
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21])) #一共21个特征，第22列是标签
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 150)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines(): #测试集
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))