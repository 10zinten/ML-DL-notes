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

