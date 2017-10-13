import os
import numpy as np
import tensorflow as tf
from scipy import misc
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import EmotionDetectionUtils
from EmotionDetectionUtils import testResult

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

img = mpimg.imread(str(raw_input("Enter the image name with extension: ")))
gray = rgb2gray(img)
# plt.imshow(gray, cmap = plt.get_cmap('gray'))
# plt.show()

sess = tf.InteractiveSession()
new_saver = tf.train.import_meta_graph('EmotionDetector_logs/model.ckpt-1000.meta')
new_saver.restore(sess, 'EmotionDetector_logs/model.ckpt-1000')
tf.get_default_graph().as_graph_def()
x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")

image_test = np.resize(gray, (1, 48, 48, 1))
tResult = testResult()
num_evals = 1000

for i in range(0,num_evals):
	result = sess.run(y_conv, feed_dict={x:image_test})
	label = sess.run(tf.argmax(result, 1))
	label = label[0]
	label = int(label)
	tResult.evaluate(label)

tResult.display_result(num_evals)
