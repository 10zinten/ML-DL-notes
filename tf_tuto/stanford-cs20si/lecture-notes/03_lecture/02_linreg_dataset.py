import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_FILE = '../data/birth_life_2010.txt'

data, n_samples = utils.read_birth_life_data(DATA_FILE)

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
print(dataset.output_types)
print(dataset.output_shapes)

# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# At each iteration of X and Y
# we get different value for X and Y

# with tf.Session() as sess:
#    print(sess.run([X, Y]))
#    print(sess.run([X, Y]))
#    print(sess.run([X, Y]))

w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

Y_predicted = X * w + b

loss = tf.square(Y - Y_predicted, name='loss')
# loss = utils.huber_loss(Y, Y_predicted)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg_dataset', sess.graph)

    for i in range(100):
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass

        if i%10 == 0:
            print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()

    w_out, b_out = sess.run([w, b])
    print('w: %f, b: %f' % (w_out, b_out))

print('[INFO] Training Time: %f secs' % (time.time() - start))

plt.plot(data[:, 0], data[:, 1], 'bo', label="Real data")
plt.plot(data[:, 0], data[:, 0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()

# some usefule tf Dataset method
'''
dataset = dataset.shuffle(1000)
dataset = dataset.repeat(100)
dataset = dataset.batch(128)
dataset = dataset.map(lambda x: tf.one_hot(x, 10))
# convert each element of dataset to one_hot vector
'''

