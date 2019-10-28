# -*- coding:utf-8 -*-
# @Time    : 2019/10/14 16:20
# @Author  : Ray.X
# Tensorflow 使用笔记 以手写识别为例
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf
import sys
import time
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
mnist = read_data_sets('MNIST_data', one_hot=True)
sess = tf.compat.v1.InteractiveSession()  # 创建交互会话 Session

"""
placeholder 建立抽象模型
tf.compat.v1.placeholder(dtype, shape=None, name=None) 占位符定义
    dtype 数据类型常用的是tf.float32,tf.float64 （32占内存小，64精度更高）
    shape=None 默认None 定义数据维数 [None, 3]表示列是3，行不定
    name=None 名称
Variable    变量声明
tf.Variable(initializer, name=None) 
    initializer 初始化参数
    name=None 名称
tf.matmul(x, W) 矩阵乘法
"""

x = tf.compat.v1.placeholder(tf.float32, [None, 784])  # 定义训练数据 784组
y = tf.compat.v1.placeholder(tf.float32, [None, 10])   # 定义训练标签 10类
W = tf.Variable(tf.zeros([784, 10]))         # 权值
b = tf.Variable(tf.zeros([10]))              # 偏置

sess.run(tf.initialize_all_variables())  # 所有变量初始化


"""
tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None) 
函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    input_tensor: 待降维的tensor
    axis=None: 指定轴 0列 1行
    keep_dims=False 是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
    name： 操作的名称
     reduction_indices：在以前版本中用来指定轴，已弃用
tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和;
tf.reduce_max  :  计算tensor指定轴方向上的各个元素的最大值;
tf.reduce_all :  计算tensor指定轴方向上的各个元素的逻辑和（and运算）;
tf.reduce_any:  计算tensor指定轴方向上的各个元素的逻辑或（or运算）;

"""
a = tf.nn.softmax(tf.matmul(x, W) + b)       # 预测函数 实际输出
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)  # 梯度下降法。学习速率为0.5
train = optimizer.minimize(cross_entropy)  # 训练目标，取最小化损失函数

# 评价模型
correct_pre = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# 训练
start = time.time()
for i in range(10000):
    if i % 100 == 0:
        rate = i / 10000
        rate_num = int(rate * 50)
        r = '\rstep-%d [%s%s]\t%d%% %d/%d' % \
            (i, '>' * rate_num, '-' * (50 - rate_num), int(rate * 100), i, 10000)
        sys.stdout.write(r)
        # time.sleep(0.001)

    batch = mnist.train.next_batch(50)
    train.run(feed_dict={x: batch[0], y: batch[1]})
end = time.time()

print('\n', end-start)
print('\n', acc.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# 关闭 Session
sess.close()
