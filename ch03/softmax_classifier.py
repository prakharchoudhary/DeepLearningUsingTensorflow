import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from random import randint 

logs_path = 'log_simple_stats_softmax'
batch_size = 100
learning  = 0.5
training_epochs = 10

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name="input")
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 784])

evidence = tf.matmul(XX, W) + b
Y = tf.nn.softmax(evidence, name="output")
# The Y output matrix will be formed of 100 rows and 10 columns.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
# minimize the error function using the GD
train_step = tf.train.GradientDescentOptimizer(0.5)\
	.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(Y, 1),\
	tf.argmax(Y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,\
	tf.float32))

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	for epoch in range(training_epochs):
		batch_count = int(mnist.train.num_examples/batch_size)
		for i in range(batch_count):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			feed_dict={X: batch_x, Y_: batch_y}
			_, summary = sess.run([train_step, summary_op],
					feed_dict=feed_dict)
			writer.add_summary(summary, epoch * batch_count + i)
		print("Epoch: ", epoch)

	print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
	print("done")

	num = randint(0, mnist.test.images.shape[0])
	img = mnist.test.images[num]

	classification = sess.run(tf.argmax(Y, 1), feed_dict={X: [img]})
	print('Neural Network predicted', classification[0])
	print('Real label is:', np.argmax(mnist.test.labels[num]))

	# saving our model
	saver = tf.train.Saver()
	save_path = saver.save(sess, "softmax_mnist.ckpt")
	print("Model saved to %s" % save_path)
