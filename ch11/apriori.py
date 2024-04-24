
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet): # 构造集合C1
    C1 = []
    for transaction in dataSet: #每一行数据
        for item in transaction: # 每一个数据
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))  # use frozen set so we 将C1元素映射到frozenset，将集合做为字典健值使用
    # can use it as a key in a dict

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D: # 数据集
        for can in Ck: # 候选集
            if can.issubset(tid): #集合元素是否在数据集中
                if not can in ssCnt: ssCnt[can ] =1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = [] # 返回空列表
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key ] /numItems # 计算关键字
        if support >= minSupport:
            retList.insert(0 ,key) # 插入数据
        supportData[key] = support
    return retList, supportData #删除一个元素后的候选列，以及每个概率值

def aprioriGen(Lk, k): #creates Ck # 根据每次LK构建Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2] # 简化运算过程，最终高效得到集合
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal 前k-2个项相同时将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5): # 构建频繁项集
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1] # 构造D和C1来计算L1，相当于初始化
    k = 2
    while (len(L[k-2]) > 0): # 从L[0]依次往后推L[1],...
        Ck = aprioriGen(L[k-2], k) # 构造集合数据能够组成个线性组合
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk 扫描数据集，从CK到LK，过滤到概率小的数据
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD 关联规则
    bigRuleList = [] # 规则列表
    for i in range(1, len(L)):#only get the sets with two or more items 数据构成的每一个元素包括L[0],L[1],...
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] # 创建只含单个元素的列表
            if (i > 1): # 合并项目集
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # 计算可信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7): # 计算置信度
    prunedH = [] #create new list to return 构造新列表返回
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence计算置信度
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7): # 合并频繁项集 生成候选规则集合
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 候选规则
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

from time import sleep
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'): # 找到投票信息
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList

def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId) # 验证是否正常工作
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if vote.candidateName not in transDict: # 如果没有该政客需要将其添加哈希表中
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic': # 不同党派信息
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount) # 投票信息
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning