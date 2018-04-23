import tensorflow as tf

tf.InteractiveSession()

'''
Python Native Types:
- TF takes in py native types (boolean, numeric(intm float) and strings.
- Single values => 0-d tensor (scalar)
- List of values => 1-d tensor (vector)
- List of list => 2-d tensor (matrix)
'''

t_0 = 19 # Treated as a 0-d tensor, or "scalar"
zl = tf.zeros_like(t_0)              # ==> 0
ol = tf.ones_like(t_0)               # ==> 1
print(zl.eval(), ol.eval())

t_1 = [b"apple", b"peach", b"grap"]  # treated as a 1-d tensor, or "vector"
zl = tf.zeros_like(t_1)              # ==> [b'' b'' b'']
# ol = tf.ones_like(t_1)             # ==> TypeError, there is no 1 as string
print(zl.eval())

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]         # tread as a 2-d tensor, or "matrix"
zl = tf.zeros_like(t_2)              # ==> 3x3 tensor, all elements are False
ol = tf.ones_like(t_2)               # ==> 3x3 tensor, all elements are True
print(zl.eval())
print(ol.eval())


'''
TensorFlow Native Types:
- Checkout tf.Dtype class for list of TF dataTypes
'''


'''
NumPy Data Types:
- TF was designed to integrate seamlessly with Numpy.
- np.int32 == tf.in32 return True
'''

# passing NumPy types to TensorFlow ops
import numpy as np
a = tf.ones([2, 2], np.float32)  # ==> [[1.0 1.0] [1.0 1.0]]
print(a.eval())

''' continue:
- In tf.Session.run(), if request object is a Tensor, the output will be NumPy array.
- TL;DL: Most of the times, we can use TensorFlow types and NumPy types interchaneably.
- But always use TensorFlow types when possible.
'''
