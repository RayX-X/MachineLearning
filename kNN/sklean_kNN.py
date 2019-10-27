# -*- coding:utf-8 -*-
# @Time    : 2019/10/20 21:43
# @Author  : Ray.X
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from kNN_ import img2vector
import os
kNN = KNeighborsClassifier()


train_labels = []
train_list = os.listdir('digits/trainingDigits')  # load the training set
m = len(train_list)
train_mat = np.zeros((m, 1024))
for i in range(m):
    file_name = train_list[i]
    file_str = os.path.splitext(file_name)[0]  # take off .txt
    num = int(file_str.split('_')[0])
    train_labels.append(num)
    train_mat[i, :] = img2vector('digits/trainingDigits/%s' % file_name)

test_labels = []
test_list = os.listdir('digits/testDigits')  # iterate through the test set
m_test = len(test_list)
test_mat = np.zeros((m_test, 1024))
for i in range(m_test):
    file_name = test_list[i]
    file_str = os.path.splitext(file_name)[0]  # take off .txt
    num = int(file_str.split('_')[0])
    test_labels.append(num)
    test_mat[i, :] = img2vector('digits/testDigits/%s' % file_name)

# 训练
classifier_train = kNN.fit(train_mat, train_labels)
# 测试
classifier_predict = kNN.predict(test_mat)

probility = kNN.predict_proba(test_mat)
# 计算各测试样本基于概率的预测
neighborpoint = kNN.kneighbors(test_mat[-1:], 3, True)
# 计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
score = kNN.score(test_mat, test_labels, sample_weight=None)
# 调用该对象的打分方法，计算出准确率


print('classifier_predict =\n ', classifier_predict)
# 输出测试的结果

print('neighborpoint of last test sample:', neighborpoint)

print('probility:', probility)

print('Accuracy:', score)
# 输出准确率计算结果