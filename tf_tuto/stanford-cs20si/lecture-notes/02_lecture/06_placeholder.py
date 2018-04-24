import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Old way: placeholder and feed_dict
    - We can assemble the graphs frist w/o knowing the values need for the
      computation with placeholder.
'''

# tf.placeholder(dtype, shape=None, name=None)
a = tf.placeholder(tf.float32, shape=[3]) # a is placeholder for a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# print(tf.Graph.is_feedable(b))

c = a + b # use the placeholder as you would any tensor
with tf.Session() as sess:
    writer = tf.summary.FileWriter('graphs/placeholders', tf.get_default_graph())
    # compute the value if c given the a [1, 2, 3]
    print(sess.run(c, {a: [1, 2, 3]}))   # [6. 7. 8.]
    writer.close()

# Any Tensors that are feedable can be fed
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    print(sess.run(b))                    # ==> 21
    print(sess.run(b, feed_dict={a: 15})) # ==> 45
