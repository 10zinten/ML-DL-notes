import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
- Constant's value is stored in the graph replicate whenever the graph is
loaded.

Graph Definition => print graph's protobuf
'''

my_const = tf.constant([1.0, 2.0], name="my_const")
# print graph's protobuf
# print(tf.get_default_graph().as_graph_def())


'''
Creating variables
- tf.contant() 'c' is lowercase and tf.Variable() 'V' is uppercase.
- It's because tf.contant is an op and tf.Variable() is a class with multiple ops.

x = tf.Variable(...)
x.initializer # init
x.value() # read op
x.assign(...) # wrtie op
x.assign_add(...) #
'''

# old way of creating variable - tf.Variable()
so = tf.Variable(2, name="scalar")
mo = tf.Variable([[0, 1], [2, 3]], name="matrix")
wo = tf.Variable(tf.zeros([784, 10]))

# New way with wrapper tf.get_variable()
sn = tf.get_variable("scalar", initializer=tf.constant(2))
mn = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
wn = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())


'''
Initialize variable
- Run into "FailedPreconditionError" when trying to evaluate the variable before
  initializing them.
- Three ways:
    - Initialize all the variables at once.
    - Initialize only subset of variables.
    - Initialize each varaible separately.
    - Inintialse a variable by loading its value from a file => later topic
'''

with tf.Session() as sess:
    # initilizing all variable at once
    # sess.run(tf.global_variables_initializer())

    # initialize only subset of variable
    sess.run(tf.variables_initializer([mo, mn]))

    # initialze each variable separately
    sess.run(sn.initializer)

    print(sess.run([mo, mn, sn]))


'''
Assign values to variable
- Initializer op is an assign op that assigns the variable's initial value to
  the variable itself, so No need to initialize variable if its being assigned.

- For incrementing and Decrementing:
    - tf.Variable.assign_add() & tf.Variable.assign_sub() methods
    - since these ops depend on the initial values of the variable, so need to
      intialize
'''

# tf.Variable.assign()
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)    # assign_op initialized the W
    print(W.eval()) # >> 100

# Interesting example
# create a variable whose original value is 2
a = tf.get_variable('scalar_1', initializer=tf.constant(2))
a_times_two = a.assign(a * 2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(a_times_two)
    sess.run(a_times_two)
    sess.run(a_times_two)
    print(a.eval())

# Incrementing and decrementing ops
W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)  # initialize required
    print(sess.run(W.assign_add(10))) # >> 20
    print(sess.run(W.assign_sub(2)))  # >> 18

# Each Session can have its own current value for a variable defined in a graph
# Because TF session maintian values separately.
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))    # >> 20
print(sess2.run(W.assign_sub(2)))    # >> 8
print(sess1.run(W.assign_add(100)))   # >> 120
print(sess2.run(W.assign_sub(50)))    # >> -42
sess1.close()
sess2.close()

# Initialzing dependent variable, U = W * 2
W = tf.Variable(tf.truncated_normal([700, 10]))
# U = tf.Variable(W * 2)

# use initialized_value() to make sure W is initialized before its value is used
# to initialize U.
U = tf.Variable(W.initialized_value() * 2)


'''
Contorl Dependencies:
    We can specify which ops should be run first.
    Use tf.Graph().control_depencies([control_inputs])

    # Your graph g have 5 ops: a, b, c, d, e
    with g.control_dependencies([a, b, c]):
        # `d` and `e` will only run after `a`, `b` and `c` have executed.
        d = ...
        e = ...
