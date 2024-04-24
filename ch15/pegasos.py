
def predict(w, x):
    return w*x.T

def batchPegasos(dataSet, labels, lam, T, k):
    m,n = shape(dataSet); w = zeros(n);
    dataIndex = range(m)
    for t in range(1, T+1): # 迭代次数
        wDelta = mat(zeros(n)) #reset wDelta
        eta = 1.0/(lam*t) # lam是参数,t是训练次数，eta为学习率逐渐变小
        random.shuffle(dataIndex) # 数据集打乱
        for j in range(k):#go over training set 遍历数据集 待处理数据大小
            i = dataIndex[j]
            p = predict(w, dataSet[i,:])        #mapper code
            if labels[i]*p < 1:                 #mapper code
                wDelta += labels[i]*dataSet[i,:].A #accumulate changes
        w = (1.0 - 1/t)*w + (eta/k)*wDelta       #apply changes at each T
    return w