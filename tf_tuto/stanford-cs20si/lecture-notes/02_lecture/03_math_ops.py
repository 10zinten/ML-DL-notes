import tensorflow as tf


'''
TensorFlow’s zillion operations for division.

tf.div - TF style division
tf.division  - Python's style division
'''

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))         # ==> [[0 0] [1 1]]
    print(sess.run(tf.divide(b, a)))      # ==> [[0. 0.5] [1. 1.5]]
    print(sess.run(tf.truediv(b, a)))     # ==> [[0. 0.5] [1. 1.5]]
    print(sess.run(tf.floordiv(b, a)))    # ==> [[0 0] [1 1]]
    # print(sess.run(tf.realdiv(b, a)))   # ErrorL only works for real values
    print(sess.run(tf.truncatediv(b, a))) # ==> [[0 0] [1 1]]
    print(sess.run(tf.floor_div(b, a)))   # ==> [[0 0] [1 1]]


'''
tf.add_n: Allow to add multiple tensors.

tf.add_n([1, b, c] => a + b + c
'''
a = tf.constant(1, name="a")
b = tf.constant(2, name="b")
c = tf.constant(3, name="c")
add_n = tf.add_n([a, b, c])
with tf.Session() as sess:
    print(sess.run(add_n))


'''
Dot product:
    tf.matmul no longer does dot product. It multiplies matrices of ranks greater or equal to 2.
    Use tf.tensordot
'''
a = tf.constant([10, 20], name="a")
b = tf.constant([2, 3], name="b")
with tf.Session() as sess:
    print(sess.run(tf.multiply(a, b)))      # ==> [20 60]
    print(sess.run(tf.tensordot(a, b, 1)))  # ==> 80
