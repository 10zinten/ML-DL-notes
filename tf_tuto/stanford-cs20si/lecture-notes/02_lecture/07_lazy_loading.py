import tensorflow as tf

# Normal Loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
    for _ in range(10):
        sess.run(z)
    writer.close()

# lazy loading
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
    for _ in range(10):
        sess.run(tf.add(x, y))  # someone decides to be clever to save one line of code.
    print(tf.get_default_graph().as_graph_def())  # to print graph definition.
    writer.close()
