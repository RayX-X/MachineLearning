# -*- coding:utf-8 -*-
# @Time    : 2019/10/18 20:16
# @Author  : Ray.X
import numpy as np
import operator
from os import listdir, path
import sys
import time


def create_data():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def classify_data(indata, traindata, labels, k):
    """
    :param indata: 测试数据
    :param traindata: 训练数据
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
    data_size = traindata.shape[0]
    diff_mat = np.tile(indata, (data_size, 1)) - traindata
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    short_diseances = np.argsort(distances)
    # 以上计算距离 以下选取距离最小的k个点
    count = {}
    for i in range(k):
        vote_label = labels[short_diseances[i]]
        count[vote_label] = count.get(vote_label, 0) + 1
    # 排序
    short_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return short_count[0][0]


def img2vector(filename):
    """
    创建1x1024 (32x32) 的数组 将数据传入数组组成一行向量
    :param filename: 文件地址
    :return: 向量
    """
    vec = np.zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        line_str = f.readline()
        for j in range(32):
            vec[0, 32 * i + j] = int(line_str[j])
    return vec


def handwriting():
    """
    手写识别
    :return:
    """
    labels = []
    train_list = listdir('digits/trainingDigits')  # load the training set
    m = len(train_list)
    train_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_list[i]
        file_str = path.splitext(file_name)[0]  # take off .txt
        num = int(file_str.split('_')[0])
        labels.append(num)
        train_mat[i, :] = img2vector('digits/trainingDigits/%s' % file_name)

    test_list = listdir('digits/testDigits')  # iterate through the test set
    error_count = 0.0
    m_test = len(test_list)
    for i in range(m_test):
        file_name = test_list[i]
        file_str = path.splitext(file_name)[0]  # take off .txt
        num = int(file_str.split('_')[0])
        test_mat = img2vector('digits/testDigits/%s' % file_name)
        classifier_result = classify_data(test_mat, train_mat, labels, 3)

        super_print(classifier_result, num, i)

        time.sleep(0.001)
        if classifier_result != num:
            print("\r错误的分类器结果: %d, 实际值: %d， 第%d次测试" % (classifier_result, num, i))
            error_count += 1.0

    print("\n错误的次数: %d" % error_count, "错误率: %f" % (error_count / float(m_test)))


def super_print(classifier_result, num, i):
    """
    这个输出比较好看 也能用于输出训练过程
    :param classifier_result: 分类器分类结果
    :param num: 实际值
    :param i: 第几次
    :return:
    """
    r = "\r分类器结果: %d, 实际值: %d， 第%d次测试" % (classifier_result, num, i)
    # r = '\rstep-%d loss value-%.4f[%s%s]\t%d%% %d/%d' % \
    # (step, loss, '>' * rate_num, '-' * (40 - rate_num), int(rate * 100), step, total)
    sys.stdout.write(r)
    sys.stdout.flush()  # 刷新 windows 下可有可无 Lunix 下必须要有

    
if __name__ == "__main__":
    handwriting()
