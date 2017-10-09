import numpy as np
import tensorflow as tf

# input value
input_value = tf.constant(0.5, name="input_value") 

# weight
weight = tf.Variable(1.0, name="weight")

# expected_value
expected_output = tf.constant(0.0, name="expected_output")

# model or output computation
model = tf.multiply(input_value,weight,"model")

# loss function
loss_function = tf.pow(model - expected_output, 2, "loss_function")

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(loss_function)

for value in [input_value, weight, expected_output, model, loss_function]:
	tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()
sess = tf.Session()
summary_writer = tf.summary.FileWriter('log_simple_stats',sess.graph)

sess.run(tf.global_variables_initializer())
for i in range(100):
	summary_writer.add_summary(sess.run(summaries), i)
	sess.run(optimizer)
