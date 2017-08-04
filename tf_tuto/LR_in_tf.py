import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure start
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = tf.add(tf.multiply(weights, x_data), biases)

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_data)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer() # important

# start the session
sess = tf.Session()
sess.run(init)

for step in range(201):
  sess.run(train)
  if step % 20 == 0:
    print(step, sess.run(weights), sess.run(biases))

sess.close()
