import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'

import utils

DATA_FILE = '../data/birth_life_2010.txt'

# Huber error
def huber_loss(labels, predictions, delta=14.0):  # found in utils.py
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

# For reusing the weight and bias variabl
def get_scope_variable(scope, var, shape=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            var = tf.get_variable(var, initializer=tf.constant(0.0), shape=shape)
    return var

def model(loss_type="MSE"):

    # Step 1: read in data
    data, n_samples = utils.read_birth_life_data(DATA_FILE)

    # Step 2: create placeholders for X (birth_rate) and Y (life expectancy)
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    # Step 3: create weight and bias, initialize to 0
    w = get_scope_variable('linreg', 'wieghts')
    b = get_scope_variable('linreg', 'bias')

    # Step 4: build the model to predict Y
    Y_predicted = w * X + b

    # Step 5: use the squared error as the loss function
    # Mean squared error
    if loss_type == "MSE":
        print("[INFO] Training with Square Loss")
        loss = tf.square(Y - Y_predicted, name='loss')
        writer = tf.summary.FileWriter('./graphs/linear_reg_MSE', tf.get_default_graph())
    else:
        print("[INFO] Training with Huber Loss")
        loss = huber_loss(Y, Y_predicted)
        writer = tf.summary.FileWriter('./graphs/linear_reg_HUBER', tf.get_default_graph())

    # Step 6: Use GD t minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    start = time.time()
    with tf.Session() as sess:
        # Step 7: initialze the necessary variables, in this case W and b
        sess.run(tf.global_variables_initializer())

        # Step 8: Train the model for 100 epochs
        for i in range(100):
            total_loss = 0
            for x, y in data:
                # Session execute optimizer and fetch value of loss
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
                total_loss += l

            if i % 10 == 0:
                print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

        # close the writer when done
        writer.close()

        # Step 9: output the value of W and b
        w_out, b_out = sess.run([w, b])

    print('[INFO] Training Time: %f seconds' % (time.time() - start))
    print()

    return w_out, b_out

# Compare Square loss and Huber loss
MSE_w_out, MSE_b_out = model()  # MSE by default
Huber_w_out, Huber_b_out = model(loss_type="Huber")


data, n_samples = utils.read_birth_life_data(DATA_FILE)

# plot the results
plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * MSE_w_out + MSE_b_out, 'r', label='Predicted data with square error')
plt.plot(data[:, 0], data[:, 0] * Huber_w_out + Huber_b_out, c='g', label='Predicted data with Huber loss')
plt.legend()
plt.show()
