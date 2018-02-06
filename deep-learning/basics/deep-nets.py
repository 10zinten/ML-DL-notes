import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# about one_hot parameter
# we have 10 classes, 0-9, with one_hot as true, each data point looks like following
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''

# hidden layer and their #units
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10

# Python optimisatioin variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders

# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# output data placeholder - 10 digist/classes
y = tf.placeholder(tf.float32, [None, n_classes])

# Nueral network Models
def neural_networl_model(data):

  # initialization of hidden layers and output layer

  hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1], stddev=0.03), name='W1'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]), name='b1')}

  hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.03), name='W2'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]), name='b2')}

  hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl1], stddev=0.03), name='W3'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]), name='b3' )}

  output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

  # calculate the output of the hidden layer
  #(input_data * wieghts) + baises

  l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
  l1 = tf.nn.relu(l1)

  l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
  l2 = tf.nn.relu(l2)

  l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
  l3 = tf.nn.relu(l3)

  output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

  return output


# Setting up training process

def train_neural_network(x):
  prediction = neural_networl_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)

  hm_epochs = 10

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
      epoch_loss = 0
      for _ in range(int(mnist.train.num_examples/batch_size)):
        epoch_x, epoch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c
      print('Epoch: ', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)


    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)



