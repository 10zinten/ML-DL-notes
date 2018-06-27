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

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()  # call this at program start-up

x = [[2.]]   # No need for placeholder!
m = tf.matmul(x, x)

print(m)  # No session!
# tf.Tensor([[4.]], shape=(1, 1), dtype=float32)
