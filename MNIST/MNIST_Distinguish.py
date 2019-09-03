'''
Creat on May 18, 2018

@author CZH
'''

'''
    MNIST（Mixed National Institute of Standards and Technology database）是一个入门级的计算机视觉数据集,其中包含各种手写数字图片
    MNIST数据集被分为三部分：55000个训练样本（mnist.train),5000个验证集（mnist.validation),10000个测试样本（mnist.test)
    x为一个大小为28*28的手写数字图片像素值矩阵,标签y是第n维度的数字为1的10位维向量。例如，标签3用one—hot向量表示为[0,0,0,1,0,0,0,0,0,0]
    数据都已经进行过标准化处理

'''

#导入要用到的库
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True) #导入数据集

#初始化参数
W = tf.Variable(tf.zeros([mnist.train.images.shape[1],10])) #W初始化为0
b = tf.Variable(tf.zeros([10])) #b初始化为0
costs = []

#建立模型
x = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]])
y = tf.placeholder(tf.float32, [None, 10]) #建立训练集占位符

y_hat = tf.nn.softmax(tf.matmul(x,W) + b) #softmax激活
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1])) #成本函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost) #梯度下降，最小化成本

sess = tf.InteractiveSession() #创建session
tf.global_variables_initializer().run() #初始化变量（声明了变量，就必须初始化才能用）

#迭代运算
for epoch in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #每次使用100个小批量数据
    sess.run([train_step, cost], feed_dict = {x: batch_xs, y: batch_ys}) #进行训练

#计算准确率
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))