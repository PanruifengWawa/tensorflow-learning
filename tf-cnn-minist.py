from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os
import tempfile
import numpy
from six.moves import urllib
from six.moves import xrange

import tensorflow as tf

###读取数据
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True) 


x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

###初始化
#权重的初始化，正太分布，mean=0，s=0.1
def weight_variable(shape):
    weight = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weight)

#偏置项初始化，全置为0.1
def bias_constant(shape):
    bias = tf.constant(0.1, shape = shape)
    return tf.Variable(bias)

####↑面所有的初始值，可以随意定，避免出现全0

###定义卷积和池化
#卷积
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
###

###第一层
#卷积，卷积核：5*5的大小，输入单通道、1个特征，输出32个特征
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_constant([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
###

###第二层
#卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_constant([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
###

###全连接
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_constant([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#Dropout，防止过拟合，对于某个神经元，以p的概率被采用
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
###

###输出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_constant([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


###训练和评估
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
cross_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuray = tf.reduce_mean(tf.cast(cross_prediction,"float"))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuray,feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print ( train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test")
    print (sess.run(accuray,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print("finish1")