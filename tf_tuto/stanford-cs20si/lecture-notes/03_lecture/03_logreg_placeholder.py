import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils

# Define hyperparameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30


# Step 1: Read in data
# using TF Lean's built in function to load MINST data to the folder ../data/mnist\
mnist = input_data.read_data_sets('../data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

# Step 3: create weights and bias
# weights is initialized to random variables with mean of 0, stdev of 0.01
# bias is initialized to 0
# Shape of weights depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# Shape of b depends on Y
w = tf.get_variable(name='weights', shape=(784 , 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch
# loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.log(Y), reduction_indices=[1]))

tf.summary.scalar('loss', loss)

activations = tf.nn.softmax(logits)
tf.summary.histogram('activations', activations)

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

# Merge all the summaries and write the out to logdir
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())
start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)

    # train the model n_epochs times
    for i in range(n_epochs):
        total_loss = 0
        counter = 0

        for n in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch, summary = sess.run([optimizer, loss, merged], {X: X_batch, Y:Y_batch})
            total_loss += loss_batch

            # Write logs at every iteration
            writer.add_summary(summary, i * n_batches+n)

        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('[INFO] Training Time: {0} secs'.format(time.time() - start_time))

    # Test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0

    for _ in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
        total_correct_preds += accuracy_batch

    print('[INFO] Acuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

writer.close()
