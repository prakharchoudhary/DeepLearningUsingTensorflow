import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# configuring network parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# sizes of hidden features
n_hidden_1 = 256
n_hidden_2 = 128

# size of input images
n_input = 784

X = tf.placeholder("float", [None, n_input])

weights = {
	'encoder_h1': tf.Variable\
	(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable\
	(tf.random_normal([n_hidden_1, n_hidden_2])),
	'decoder_h1': tf.Variable\
	(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h2': tf.Variable\
	(tf.random_normal([n_hidden_1, n_input])),
}

biases = {
	'encoder_b1': tf.Variable\
	(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable\
	(tf.random_normal([n_hidden_2])),
	'decoder_b1': tf.Variable\
	(tf.random_normal([n_hidden_1])),
	'decoder_b2': tf.Variable\
	(tf.random_normal([n_input])),	
}

# autoencoder layers
encoder_in = tf.nn.sigmoid(tf.add(
	tf.matmul(X,\
		weights['encoder_h1']), biases['encoder_b1'])
)

encoder_out = tf.nn.sigmoid(tf.add(
	tf.matmul(encoder_in,\
		weights['encoder_h2']), biases['encoder_b2'])
)

decoder_in = tf.nn.sigmoid(tf.add(
	tf.matmul(encoder_out,\
		weights['decoder_h1']), biases['decoder_b1'])
)

decoder_out = tf.nn.sigmoid(tf.add(
	tf.matmul(decoder_in,\
		weights['decoder_h2']), biases['decoder_b2'])
)

# error minimization
y_pred = decoder_out
y_true = X
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

# here goes the graph! let's roll
with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)

	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_xs, batch_ys = \
				mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

		if epoch % display_step == 0:
			print("Epoch: "+"%04d" % (epoch+1)+\
				" cost="+"{:.9f}".format(c))
	print("Optimization FInished!")
	encode_decode = sess.run(
		y_pred, feed_dict=\
		{X: mnist.test.images[:examples_to_show]})

	f, a = plt.subplots(2, 4, figsize=(10, 5))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	f.show()
	plt.draw()
	plt.show()


