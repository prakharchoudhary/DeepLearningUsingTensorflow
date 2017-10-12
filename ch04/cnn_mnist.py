import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256
img_size = 28
num_classes = 10

# placeholders for data
X = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
Y = tf.placeholder(tf.float32, [None, num_classes])

# download and extract mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
	mnist.test.images, mnist.test.labels

# reshape data acc. to model
trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)

# initializing the weights
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.1))

# convolutional model
def model(X, w, w2, w3, w4, w_0, p_keep_conv, p_keep_hidden):
	conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')
	conv1_a = tf.nn.relu(conv1)
	conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv1 = tf.nn.dropout(conv1, p_keep_conv)

	conv2 = tf.nn.conv2d(conv1, w2, strides=[1, 1, 1, 1], padding='SAME')
	conv2_a = tf.nn.relu(conv2)
	conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	conv2 = tf.nn.dropout(conv2, p_keep_conv)

	conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1], padding='SAME')
	conv3_a = tf.nn.relu(conv3)

	FC_layer = tf.nn.max_pool(conv3_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])
	FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)

	output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
	output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

	result = tf.matmul(output_layer, w_0)
	return result

w = init_weights([3, 3, 1 , 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_0 = init_weights([625, num_classes])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_0, p_keep_conv, p_keep_hidden)

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
cost = tf.reduce_mean(Y_)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
	tf.global_variables_initializer().run()

	for i in range(100):
		training_batch = zip(range(0, len(trX), batch_size),
			range(batch_size, len(trX)+1, batch_size))
		for start, end in training_batch:
			sess.run(optimizer, feed_dict={X: trX[start:end], Y: trY[start:end],
				p_keep_conv: 0.8, p_keep_hidden: 0.5})
		test_indices = np.arange(len(teX))
		np.random.shuffle(test_indices)
		test_indices = test_indices[0: test_size]
		f_dict = {X: teX[test_indices],
			Y: teY[test_indices],
			p_keep_conv: 1.0,
			p_keep_hidden: 1.0}
		print(i, np.mean(np.argmax(teY[test_indices], axis=1)==sess.run(predict_op, feed_dict=f_dict)))
