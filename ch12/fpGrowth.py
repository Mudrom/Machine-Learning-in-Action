class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue # 名字
        self.count = numOccur #值
        self.nodeLink = None # 链表的链
        self.parent = parentNode  # needs to be updated 父母节点
        self.children = {} # 子节点

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):  # create FP-tree from dataset but don't mine
    headerTable = {}
    # go over dataSet twice
    # first time
    for trans in dataSet:  # first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans] # 统计数据出现次数
    for k in list(headerTable.keys()):  # remove items not meeting minSup
        if headerTable[k] < minSup:
            del (headerTable[k]) # 移除不满足最小支持度的元素项
    freqItemSet = set(headerTable.keys())
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  # if no items meet min support -->get out 如果没有满足元素项就退出
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # reformat headerTable to use Node link
    # print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None)  # create tree 即根节点
    # second time
    for tranSet, count in dataSet.items():  # go through dataset 2nd time
        localD = {}
        for item in tranSet:  # put transaction items in order 根据全局频率对每个事务元素排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  # populate tree with ordered freq itemset 使用排序后的频率项集对树进行填充
    return retTree, headerTable  # return tree and header table


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:  # check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count)  # incrament count 更新计数
    else:  # add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # 创建新节点
        if headerTable[items[0]][1] == None:  # update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items 迭代调用自身
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):  # this version does not use recursion 确保链表之间的链接
    while (nodeToTest.nodeLink != None):  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root 迭代上溯整颗树
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath) # 递归自己父母节点的感觉

# 遍历项目集建立条件模式基
def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath) # 上溯树
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count # 计次
        treeNode = treeNode.nodeLink # 下一节点
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]#(sort header table) 对整个表进行排序
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy() # 复制
        newFreqSet.add(basePat) # 加入集合
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet) # 频繁项集
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1]) # 条件模式基
        print('condPattBases :',basePat, condPattBases)
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print('conditional tree for: ',newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict