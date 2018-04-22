import tensorflow as tf

'''
syntax : tf.constant(value, dtype=None, shape=None, name="Const",
                      verify_shape=False)

verify_shape=True -> throws error if input shape doesn't match the shape specified.
'''

tf.InteractiveSession()

# constant of 1d tensor (vector)
a = tf.constant([2, 2], name="vector")
print(a.eval())

# constant of 2d tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="matrix")
print(b.eval())


# create tensor with specific dimension and fill it with a specific value,
# similar to numpy

''' tf.zeros(shape, dtype=tf.float32, name=None) '''
# create a tensor of shape and all elements are zeros
z = tf.zeros([2, 3], tf.int32) # ==> [[0, 0, 0], [0, 0, 0]]
print(z.eval())

''' tf.zeros_like(input_tensor, type=None, name=None, optimize=True) '''
# create a tensor of shape and type (unless type os specified) as the
# input_tensor but all elements are zeros
input_tensor = tf.constant([[0, 1], [2, 3], [4, 5]], name="input_tensor")
zl = tf.zeros_like(input_tensor) # ==> [[0, 0], [0, 0], [0, 0]]
print(zl.eval())

''' tf.ones(shape, dtype=tf.float32, name=None) '''
# create a tensor of shape and all elements are ones
o = tf.ones([2, 3], tf.int32) # ==> [[1, 1, 1], [1, 1, 1]]
print(o.eval())

''' tf.one_like(input_tensor, dtype=None, name=None, optimize=True) '''
# create a tensor of shape and type (unless type is specified) as the
# input_tensor but all the elements are ones.
ol = tf.ones_like(input_tensor)
print(ol.eval())


# Creating constants that are sequences

''' tf.lin_space(startm stop, num, name=None) '''
# create a sequence of num evenly-spaced values are generated beginning at
# start. If num > 1, the values in the sequence increase by
# (stop - start) / (num - 1), so that the last one is exactly stop.
# comparable to but slightly different from numpy.linspace
s = tf.lin_space(10.0, 13.0, 4, name="linspace") # ==> [10.0, 11.0, 12.0, 13.0]
print(s.eval())

''' tf.range([start], limit=None, delta=1, dtype=None, name='range') '''
# create a sequence of numbers that begins at start and extends by increments of
# delta up to but not included limit
# slight different from range in python
r1 = tf.range(3, 18, delta=3)
print(r1.eval())
r2 = tf.range(3, 1, delta=-0.5)
print(r2.eval())

'''
Note:
    Unlike NumPy or Python sequences, TF sequence are not iterable

    Generate randon constants from certain distributions
    tf.random_normal
    tf.truncated_normal
    tf.random_uniform
    tf.random_shuffle
    tf.random_crop
    tf.multinomial
    tf.random_gamma
    tf.set_random_seed
'''

