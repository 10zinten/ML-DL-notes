import tensorflow as tf

# tf constant
x1 = tf.constant(6)
x2 = tf.constant(5)
sum = tf.add(x1, x2)
product = tf.multiply(x1, x2)

# Create session to output
with tf.Session() as sess:
  output = sess.run(product)
  print(output)

print(output)

sess = tf.Session()
# Parameterizing graph to accept external input
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

# making computational graph more complex
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4}))

# Trainable parameters to get new output with the same input
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# Another basic Tutorial

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')

# now create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# all the constants, variables, operations and the computational graph are only created when the initialisation
# commands are run.
# setup the variable initialisation
init_op = tf.global_variables_initializer()

# The TensorFlow session is an object where all operations are run.
# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))


# create TensorFlow variables, when not shure about the input data
b = tf.placeholder(tf.float32, [None, 1], name='b') #we can inject as much 1-dimensional data that we want into the b variable of float32


