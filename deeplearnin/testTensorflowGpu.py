# -*- coding:utf-8 -*-
# @Time    : 2019/10/13 21:03
# @Author  : Ray.X
import tensorflow as tf
import datetime
#running
# Creates a graph.(cpu version)
print('cpu version')
starttime1 = datetime.datetime.now()
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6, 9], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[9, 6], name='b')
  c = tf.matmul(a, b)
  c = tf.matmul(c,a)
  c = tf.matmul(c,b)
# Creates a session with log_device_placement set to True.
sess1 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
for i in range(59999):
    sess1.run(c)
print(sess1.run(c))
sess1.close()
endtime1 = datetime.datetime.now()
time1 = (endtime1 - starttime1).microseconds
#print('time1:',time1)
#############################################
print('gpuversion')
# Creates a graph.(gpu version)
starttime2 = datetime.datetime.now()
#running
with tf.device('/gpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6, 9], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[9, 6], name='b')
  c = tf.matmul(a, b)
  c = tf.matmul(c,a)
  c = tf.matmul(c,b)
# Creates a session with log_device_placement set to True.
sess2 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
for i in range(59999):
    sess2.run(c)
print(sess2.run(c))
sess2.close()
endtime2 = datetime.datetime.now()
time2 = (endtime2 - starttime2).microseconds
print('time1:', time1)
print('time2:', time2)
