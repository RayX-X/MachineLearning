# -*- coding:utf-8 -*-
# @Time    : 2019/10/18 20:16
# @Author  : Ray.X
import numpy as np
import operator


def create_data():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def classify_data(indata, traindata, labels, k):
    """
    :param indata: 输入
    :param traindata: 原有数据
    :param labels: 分类标签
    :param k: k
    :return:
    np.tile 就是把dataSize看成一个整体，然后复制若干遍(x, (2,3))即 [[x,x,x],[x,x,x]
    array.sum(axis=0/1)  0 即普通的相加 1 为每一行向量相加
    numpy.argsort(a,axis=0/1) 对数组a排序，返回一个排序后索引，a不变 0 按行 1 按排
    shorted() 可以对列表临时排序小到大 reverse=True 则反转
    iteritems() 返回一个迭代器 items()，将一个字典以列表的形式返回。
    itemgetter函数用于获取对象的哪些维的数据，参数为一些序号 获取的不是值，而是定义了一个函数
    """
    dataSize = traindata.shape[0]
    diffMat = np.tile(indata, (dataSize, 1)) - traindata
    sq_diffMat = diffMat**2
    sq_distances = sq_diffMat.sum(axis=1)
    distances = sq_distances**0.5
    short_diseances = np.argsort(distances)
    # 以上计算距离 以下选取距离最小的k个点
    count = {}
    for i in range(k):
        voteLabel = labels[short_diseances[i]]
        count[voteLabel] = count.get(voteLabel, 0) +1
    # 排序
    short_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return short_count[0][0]


if __name__ =="__main__":
    (group, lables) = create_data()
    a = classify_data([0, 0], group, lables, 3)
    b = classify_data([1, 0], group, lables, 3)
    c = classify_data([3, 0], group, lables, 4)
    print(a)
