'''
Eager execution is:
    1) NumPy-like lib for numerical computation with support for GPU acceleration
       automatic differentiation
    2) A flexible platform for ML research and experimentation.

Key advantages:
    - compatible with Python debuggering tool: pdb.set_trace()
    - provides immediate error reporting
    - permits use of Python data structures
    - enables you to use and differentiate through Python control flow.
'''

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()  # call this at program start-up

## No boilerplate -No session, No placeholder
x = [[2.]]   # No need for placeholder!
m = tf.matmul(x, x) # matmul op provides a value immediately

print(m)  # No session!
# tf.Tensor([[4.]], shape=(1, 1), dtype=float32)


## No Lazy Loading
x = tf.random_uniform([2, 2])

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        print(x[i, j])


## Tensor Act like Numpy array
x = tf.constant([1.0, 2.0, 3.0])

# Tensors are backed by Numpy arrays
assert type(x.numpy()) == np.ndarray
squared = np.square(x) # Tensor are compatible with Numpy functions
print(squared)

# Tensors are iterable!
for i in x:
    print(i)


## Gradients - Automatic differentiation is built into eager execution
# Under the hood ...
#    - Operations are recorded on a tap
#    - The tape is Played back to compute gradients
#         - This is reverse-mode differentiation (backpropagation)

# Eg: 01
def square(x):
    return x**2

grad = tfe.gradients_function(square) # differentiation w.r.t input of square

print(square(3.))
print(grad(3.))

# Eg: 02
x = tfe.Variable(2.0) # use when eager execution is enabled
def loss(y):
    return (y - x ** 2) ** 2

grad = tfe.implicit_gradients(loss) # Differentiation w.r.t variables used to compute loss

print(loss(7.))
print(grad(7.)) # [gradient w.r.t x, x]
