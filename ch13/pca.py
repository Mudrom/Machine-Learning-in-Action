
from numpy import *
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean 10000*2
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions 去除不重要维度
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest 重组特征向量
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions # meanRemoved 10000*2 redEigVects2*1 最终得到10000*1 就投影于pca面的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 用特征向量重构新的空间再反均值化
    ## pca返回原空间 lowDDataMat原始数据在特征空间下的长度，而使用该数外积特征向量相当于将他构造新的空间，来重构符合原数据结构，但是经过压缩的数据
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
