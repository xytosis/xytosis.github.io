# this code is based off of the tutorial found at https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html

import tensorflow as tf

# initialize the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# initializes the weights with some noise
def make_weight(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

# initializes the biases with some noise
def make_bias(shape):
  return tf.Variable(tf.constant(0.1, shape = shape))

# does a convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

# max pools
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')

# here we initialize the variables
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# here we initialize the weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

W_c1 = make_weight([5, 5, 1, 32])
b_c1 = make_bias([32])

W_c2 = make_weight([5, 5, 32, 64])
b_c2 = make_bias([64])

W_f1 = make_weight([7 * 7 * 64, 1024])
b_f1 = make_bias([1024])

W_f2 = make_weight([1024, 10])
b_f2 = make_bias([10])

# reshape x
x_image = tf.reshape(x, [-1,28,28,1])

# convolve with weight tensors, add the bias and apply the ReLU neruon function, and then max pool
h_p1 = max_pool_2x2(tf.nn.relu(conv2d(x_image, W_c1) + b_c1))

# now we convolve the second layer
h_p2 = max_pool_2x2(tf.nn.relu(conv2d(h_p1, W_c2) + b_c2))

h_p2_reshape = tf.reshape(h_p2, [-1, 7*7*64])

# apply the ReLU function
h_fc1 = tf.nn.relu(tf.matmul(h_p2_reshape, W_f1) + b_f1)

# initialize dropout with probability, so we can tune it for training and testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output the softmax layer
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_f2) + b_f2)

# now finally we train
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
  batch = mnist.train.next_batch(50)
  sess.run(train_step, feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

print sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})