import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
  # add one more layer and return the output of this layer
  with tf.name_scope("layer"):
    with tf.name_scope("weights"):
      Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope("biases"):
      biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    with tf.name_scope("inputs"):
      Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

    if activation_function is None:
      outputs = Wx_plus_b
    else:
      outputs = activation_function(Wx_plus_b)
    return outputs

# make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5  + noise

# visualizing the data
# plt.scatter(x_data, y_data)
# plt.show()

# define placeholder for inputs to network
with tf.name_scope("inputs"):
  xs = tf.placeholder(tf.float32, [None, 1], name='x_inputs')
  ys = tf.placeholder(tf.float32, [None, 1], name='y_inputs')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error btw prediction and real data
with tf.name_scope("loss"):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys  - prediction), reduction_indices=[1]))
with tf.name_scope("train"):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# Tensorbroad
writer = tf.summary.FileWriter('logs/', sess.graph)

# important step
init = tf.global_variables_initializer()

sess.run(init)
