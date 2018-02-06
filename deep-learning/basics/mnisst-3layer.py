import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100
n_nodes_hl = 300

# declare the training data placeholders

# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# Declare the weights connecting the input to the hiddent layer
w1 = tf.Variable(tf.random_normal([784, n_nodes_hl], stddev=0.03), name='w1')
b1 = tf.Variable(tf.random_normal([n_nodes_hl]), name='b1')
# weights connecting the hidden layer to the ouput layer
w2 = tf.Variable(tf.random_normal([n_nodes_hl, 10], stddev=0.03), name='w2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')


# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, w1), b1)
hidden_out = tf.nn.relu(hidden_out)

# Calculate hidden layer output with softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

# calculating cost

# converting the output y_ to a clipped version, limited between 1e-10 to 0.999999
# to avoid log(0) operation occurring during training - this would return NaN and break the training process.
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

# cross entropy calculation
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))
# add an optimiser
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
  # initialize the variables
  sess.run(init_op)
  total_batch = int(len(mnist.train.labels) / batch_size)
  for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
      batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
      _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
      avg_cost += c / total_batch
    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
