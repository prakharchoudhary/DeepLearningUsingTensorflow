import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

logs_path = 'log_simple_stats_5_layers_relu_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 10

mnist = input_data.read_data_sets("MNIST_data/")

# training features and labels
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
XX = tf.reshape(X, [-1, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)

# no. of hidden neurons in each layer
L = 200
M = 100
N = 60
O = 30

# layers
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))
B1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10]))

# training
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,
				labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# using AdamOptimizer
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# maintaining logs for tensorboard
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(logs_path,
		graph=tf.get_default_graph())
	for epoch in range(training_epochs):
		batch_count = int(mnist.train.num_examples/batch_size)
		for i in range(batch_count):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			feed_dict = {X : batch_x, Y_ : batch_y}
			_, summary = sess.run([train_step, summary_op],feed_dict=feed_dict)	
			writer.add_summary(summary, epoch * batch_count + i)

		print("Epoch: ", epoch)

	print("Accuracy: ", accuracy.eval\
		(feed_dict =  {X: mnist.test.images,
					Y: mnist.test.layers}))
	print("Done")