from numpy import *


def loadDataSet(): # 构建虚拟的数据集
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet): #通过集合这个函数的性质构建一个词向量字典
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet): #遍历自己数据集合列表去看是否所有数据都存在于字典当中
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory): # 计算概率核心函数
    numTrainDocs = len(trainMatrix) # 矩阵中的行数表示输入数据集的个数
    numWords = len(trainMatrix[0]) #矩阵中的列数表示词典中单词的个数
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 计算每个类别的概率，即0+1+0+1+0+1/6
    p0Num = ones(numWords); p1Num = ones(numWords)      #根据每行的元素构建一个全1向量
    p0Denom = 2.0; p1Denom = 2.0                        #给一个原始基数2
    # 为什么要通过这样的方式构造非零向量和非零数，是为了防止在后期计算概率累积时候因为一个零值影响全局
    for i in range(numTrainDocs): # 扫描每一行
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 向量中每个元素加1
            p1Denom += sum(trainMatrix[i]) # 2+数据集的出现次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)       #change to log() 取对数是因为防止数都很小导致最终结果很小，相当于把小数据放大，避免下溢出
    p0Vect = log(p0Num/p0Denom )     #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult，内含是通过我们得到的出现向量，即出现为1，不出现为0，然后通过对应每个元素概率乘积得到最终的p1和p2概率值
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) # 计算两个概率值
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet() #读取数据
    myVocabList = createVocabList(listOPosts) #构建词典
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc)) #得到出现的向量
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) #计算每个单词属于两个类别的概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)) #变成对应的词向量
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26): #读取所有的邮件
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="windows-1252").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="windows-1252").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary构建字典
    trainingSet = list(range(50)); #构建一个索引序列
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet))) #符合均匀分布函数
        testSet.append(trainingSet[randIndex]) # 构造好测试集后就会把测试集中所有数据删除
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #变成词向量函数
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses)) # psam存储pAb
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  #通过类别的词条向量乘以概率矩阵最终得到为0或者1的概率
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText

def calcMostFreq(vocabList,fullText): #构造字典存放出现次数最多的词汇
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary']) # 对应解码获得文件数据
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1 给定标签
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary #构建词典
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words 得到次数最高的30个词汇
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0]) # 移除对应的30个数据
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex]) #获得测试集
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) # 构建trainMat词条向量
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #通过类别的词条向量乘以概率矩阵最终得到为0或者1的概率
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V