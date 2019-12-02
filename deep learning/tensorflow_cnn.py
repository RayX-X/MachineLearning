# -*- coding:utf-8 -*-
# @Time    : 2019/10/28 20:17
# @Author  : Ray.X
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
# 加载数据
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

