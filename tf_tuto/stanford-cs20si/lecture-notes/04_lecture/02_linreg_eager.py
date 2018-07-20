import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

import utils

DATA_FILE = '../data/birth_life_2010.txt'

# Enable Eager mode execution
tfe.enable_eager_execution()

# Read the data into dataset
data, n_samples = utils.read_birth_life_data(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:, 0], data[:, 1]))

# Create Variable
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# define the linear predictor
def prediction(x):
    return x * w + b

# define loss functions
def sqaure_loss(y, y_predicted):
    return (y - y_predicted) ** 2

def huber_loss(y, y_predicted, delta=1.0):
    residual = y - y_predicted
    return residual**2 if tf.abs(residual) <= delta else delta*(2*tf.abs(residual) - delta)

def train(loss_fn):
    """Train a regression model evaluated using `loss_fn`."""
    print('[INFO] Training; loss funtion: ' + loss_fn.__name__)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # define the funtion through which to defferentiate
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))

    # grad_fn(x_i, y_i) return:
    #    (1) the value of `loss_for_example` evaluated at x_i, y_i.
    #    (2) the gradients of any variables used in calculating it.
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

    start = time.time()
    for epoch in range(100):
        total_loss = 0.0
        for x_i, y_i in tfe.Iterator(dataset):
            loss, gradients = grad_fn(x_i, y_i)
            # Take an optimization step and update variables.
            optimizer.apply_gradients(gradients)
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {0}: {1}'.format(epoch, total_loss/n_samples))

    print('[INFO] Training Time: {} secs'.format(time.time() - start))


# Train the model
train(huber_loss)

# prediciton
print(prediction([[1]]).numpy())

plt.plot(data[:, 0], data[:, 1], 'bo')
plt.plot(data[:, 0], data[:, 0] * w.numpy() + b.numpy(), 'r',
         label='Huber regression')
plt.legend()
plt.show()
