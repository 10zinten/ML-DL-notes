from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])  # 2d Tensor None - Batch_size 784 - 24X24
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 2d Tensor None - Batch_size  10 - one-hot 10d vector

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5,  # padding_height
                           5,  # Padding width
                           1,  # No. of input channel
                           32, # No. of output channel
                          ])
b_conv1 = bias_variable([32])  # Bias for each output channel


# To apply layer, we first reshape x to a 4d Tensors
   # 1st dimension is number of images - None
   # 2nd and 3rd dimensions corresponding to the image width and height
   # last dimension - No. color channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve x_image wtih the weight tensor and add the bias, apply ReLU function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# ## Fully Connected layer
# Now Image size reduced to 7x7.
# We add a fully connected layer with 1024 neuron.fc

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## Learning Rate
learning_rate = 1e-4

## Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


# ## Training and Evaluating Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
batch_size = 100
costs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        epoch_cost = 0
        num_batches = mnist.train.num_examples/batch_size - 400.0
        for _ in range(int(num_batches)):
            batch = mnist.train.next_batch(50)
            _, minibatch_cost = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1]})#, keep_prob: 0.5})
            epoch_cost += minibatch_cost/num_batches
        if epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save parameters in a Varaible
    #parameters = sess.run(parameters)

    # Calculate the correct prediction
    correct_prediction = tf.equal(tf.argmax(y_conv), tf.argmax(y_))

    # calculate the correction on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Train accuracy ', accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
    print('test accuracy ', accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
