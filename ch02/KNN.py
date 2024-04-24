#!/usr/bin/env python
# coding: utf-8


from numpy import*
#这种导入方式可以直接使用 NumPy 库中的所有函数而无需加前缀 np。
#这种方法将 NumPy 库中的所有公共函数、类和变量直接导入到当前的命名空间中。因此，您可以直接使用 tile 而不是 np.tile。
#但不同的库可能有功能相似或相同名字的函数，所以不要这样同时导入很多个库，而且代码的可读性会下降。
import operator

#测试用
def createDataSet(): 
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k): #它接收四个参数：inX（要分类的数据点：就是正在被使用knn的观测点），dataSet（用于分类的数据集），labels（数据集中每个数据点的标签），和 k（在k-NN算法中选择的最近邻居的数量）
    
    dataSetSize = dataSet.shape[0]
    #这行代码获取数据集的大小。dataSet.shape[0] 返回数据集中的行数，即数据点的数量。
    #dataSet 通常是一个二维数组，其中每一行代表一个数据点，每一列代表一个特征。例如，如果数据集包含了100个样本，每个样本有5个特征，那么 dataSet 将是一个 100x5 的数组。
    #shape 是一个描述数组维度的属性。对于一个二维数组，shape 返回一个元组，其中包含两个元素：第一个元素是行数（代表数据点的数量），第二个元素是列数（代表每个数据点的特征数量）。
    #基于前面的例子，在代码 dataSet.shape[0] 中，.shape 返回的元组例如可能是 (100, 5)，表示数组有 100 行和 5 列。通过获取这个元组的第一个元素（即使用 [0] 索引），您能得到行数，也就是数据集中数据点的总数。
    
    diffMat = tile(inX, (dataSetSize,1)) - dataSet 
    #dataSet中有很多个样本，每个样本有很多个特征；inX也有相同个数的特征，但这只是一个样本。这句代码就是让inX复制到可以和dataSet中的每个样本的每个特征相减求差，以便在之后计算inX和每个dataSet中样本的距离以确定 inX 与哪些样本最相似。
    #tile的用法举例：
    #inX = np.array([1, 2, 3])
    #repeated_inX = np.tile(inX, (4, 1))
    #[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]

    sqDiffMat = diffMat**2 
    #sqDiffMat应该是一个形状和dataSet相同的数组, 如果 diffMat 的形状是 m×n（其中m是样本数，n是特征数），那么 sqDiffMat 也将是 m×n 形状
    
    sqDistances = sqDiffMat.sum(axis = 1)
    #这行代码对每一行的平方和进行求和，得到每个数据点与输入点 inX 之间的平方距离。
    distances = sqDistances**0.5 #这行代码取平方距离的平方根，得到实际的欧氏距离。
    #这两个都是一维数组。注意，二维数组的两个维度是行和列，而三维数组可以想象成一个立方体，依此类推。
    
    sortedDisIndicies = distances.argsort() #这行代码对距离进行排序，并返回排序后的索引。这些索引用于确定最近的邻居。
    #举个例子解释argsort的用法：arr = np.array([3, 1, 2])
    #sorted_indices = arr.argsort()
    #sorted_indices = [1, 2, 0] 这意味着最小的元素（1）在原数组中的索引是 1，其次是元素 2 在索引 2，最后是元素 3 在索引 0。
    #也就是说argsort按照从小到大进行排序，最左显示最小值在原数组中的索引
    #arg：这通常是指 "argument"，在编程中经常用来表示函数参数或者数据的索引（位置）。

    classCount = {} #这行初始化一个空字典，用于存储每个类别的投票计数。
    for i in range(k): #从0遍历到 k-1，用于累计最近的 k 个邻居的类别投票。
        voteIlabel = labels[sortedDisIndicies[i]]
        #在循环中，这行代码找到第 i 个最近邻居的标签。
        #labels是这个函数的输入项，也就是说user需要自带类别。所以这里输入的是labels[0]到labels[k-1]，也就是sortedDisIndicies为0到k-1的labels。
        #所以 voteIlabel 变量存储了：根据距离排序，第 i 个最近的样本属于哪个类别。注意，voteIlabel 可能会多次获取同一类别的标签！
        
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #这行代码更新 classCount 字典，将 voteIlabel 类别的计数加1。
        #classCount.get(voteIlabel, 0)：这部分是从 classCount 字典中获取 voteIlabel 类别的当前计数。如果这个类别还没有被计数过（即在字典中不存在），那么就返回 0。
        # +1：无论 voteIlabel 类别是否已存在于 classCount 字典中，都在其当前计数上加一。这代表当前考虑的邻居对该类别“投了一票”。
        
        #初始化字典：创建一个空字典：my_dict = {} 或 my_dict = dict()
        #给字典赋值：如果键不存在于字典中，赋值会创建一个新的键值对。例：my_dict[key] = value，这行代码会将键 key 与值 value 相关联并存储在字典中。
        #修改字典中的值：如果键已经存在于字典中，赋值会更新该键的值。例如：my_dict[key] = new_value
        #检查键是否存在并赋值：有时，您可能不确定一个键是否已存在于字典中。在这种情况下，可以使用 get 方法来安全地处理赋值和更新。
            #my_dict.get(key, default) 会返回字典中 key 的值；如果 key 不存在，它会返回 default。
            #因此，my_dict[key] = my_dict.get(key, 0) + 1 这行代码会检查 key 是否存在于 my_dict 中。如果存在，它会获取该键的值，加上 1，然后再将这个新值赋回给 key。如果 key 不存在，它会从 0 开始计数，并将 1 赋给 key。
         
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) #这行代码根据类别的投票数对它们进行排序，最高投票数的类别排在最前面。
    # sorted() 函数用于对所有可迭代的对象进行排序，例如列表、元组、字典等，并返回一个新的排好序的列表。
        #用法：sorted(iterable, key=None, reverse=False)
        #iterable 是要排序的可迭代对象，比如列表、元组。可迭代的意思是能够遍历，也就是里面得有好几个东西。只有一个是没办法排序的。
            #字典有点特殊的是，虽然字典本身是可迭代的，但它迭代的是键，而不是键值对。要迭代键值对，可以使用 items() 方法，它返回一个类似于列表的对象，其中每个元素都是一个元组。
        #key 是一个函数，用于从每个元素中提取一个用于比较的键（例如，按长度、某个属性等进行排序）。
            #operator.itemgetter(1)：这个函数创建了一个获取器（retriever），用于从每个元组中获取索引为 1 的元素。在一个 (key, value) 对中，索引 0 对应 key（类别名），而索引 1 对应 value（计数）。因此，itemgetter(1) 用于获取每个键值对中的计数值。这个1不是第1列，只是固定要求。
        #reverse 是一个布尔值。如果设置为 True，则列表以降序排列。
    
    return sortedClassCount[0][0] #这行代码返回投票数最多的类别标签。
    #sortedClassCount类似：[('类别B', 15), ('类别A', 10), ('类别C', 5)]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines() #读取文件的每一行，并将这些行作为一个列表存储在 arrayOLines 中。
    numberOfLines = len(arrayOLines) #计算文件中的行数，numberOfLines是int
    returnMat = zeros((numberOfLines,3)) #创建一个大小为 numberOfLines 行，3 列的零矩阵 returnMat。这个矩阵用于存储文件中的数值数据。
    #3是因为每行数据由三个数值组成，不知道是几个就先看一下
    classLabelVector = [] #初始化一个空列表 classLabelVector，用于存储每行数据的类别标签。
    index = 0 #初始化一个变量 index，用于在循环中追踪当前处理的行数。
    for line in arrayOLines:
        line = line.strip() #移除每行字符串首尾的空白字符（包括换行符）。
        # 如果不用strip的话，下面通过\t划分之后，每行的最后一个元素就会带着\n，这样数据就不一致了。
        listFromLine = line.split('\t') # 通过Tab划分每行字符串；listFromLine是列表。
        returnMat[index,:] = listFromLine[0:3] #将 listFromLine 列表中的前三个元素（预期是数值）赋值给 returnMat 矩阵的当前行。
        # 注意 listFromLine 是上面循环得到的，所以并不是returnMat的每一行都是同样的3个元素
        classLabelVector.append(int(listFromLine[-1])) 
        #将 listFromLine 列表中的最后一个元素（预期是类别标签）转换为整数，并添加到 classLabelVector 列表中。
        index += 1
    return returnMat,classLabelVector #函数返回两个对象：returnMat（一个数值矩阵）和 classLabelVector（一个类别标签的列表）。


    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
    
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    
    
    
    





