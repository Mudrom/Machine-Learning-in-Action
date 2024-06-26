

from numpy import *
from numpy import linalg as la
 #测试程序
# def loadExData():
#     return[[1, 1, 1, 0, 0],
#            [2, 2, 2, 0, 0],
#            [1, 1, 1, 0, 0],
#            [5, 5, 5, 0, 0],
#            [1, 1, 0, 2, 2],
#            [0, 0, 0, 3, 3],
#            [0, 0, 0, 1, 1]]
 #程序4-2
def loadExData():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def standEst(dataMat, user, simMeas, item):
       n = shape(dataMat)[1] # 物品数量
       simTotal = 0.0;
       ratSimTotal = 0.0
       for j in range(n):
              userRating = dataMat[user, j]
              if userRating == 0: continue
              overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                            dataMat[:, j].A > 0))[0] # 寻找两个用户都评价的物品
              if len(overLap) == 0: #如果两个用户没有重复项目
                     similarity = 0
              else:
                     similarity = simMeas(dataMat[overLap, item], \
                                          dataMat[overLap, j]) #相似度评判标准
              print('the %d and %d similarity is: %f' % (item, j, similarity))
              simTotal += similarity
              ratSimTotal += similarity * userRating
       if simTotal == 0:
              return 0
       else:
              return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)  # 奇异值分解
    Sig4 = mat(eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix 构造新的Σ 即奇异值对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items 通过svd分解前四个奇异值去重构特征空间
    print('xformedItems',xformedItems.shape)
    print('dataMat',dataMat.shape)
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, \
                             xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item) # 度量用户之间的相似度
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N] # 找到最相关的n个

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end ='')
            else:
                print(0,end ='')
        print(' ')

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV))) #构造全零矩阵
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k] #对角线赋值
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
