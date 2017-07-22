import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Python
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
x = tf.constant(100)
print(sess.run(hello))
# type(x)

sess = tf.Session()
sess.run(hello)
# type(sess.run(x))

# TensorFlow Operations
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print('Operations with constant:')
    print('Addition', sess.run(x + y))
    print('Subtraction', sess.run(x - y))
    print('Division', sess.run(x / y))
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
sub = tf.subtract(x, y)
mul = tf.multiply(x, y)

with tf.Session as sess:
    print('Operation with placeholder:')
    print('Addition:', sess.run(add, feed_dict={x: 20, y: 30}))
    print('Subtraction:', sess.run(sub, feed_dict=d))
    print('Multiplicaiton:', sess.run(mul, feed_dict=d))

a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_mult = tf.matmul(mat1, mat2)

with tf.Session as sess:
    result = sess.run(matrix_mult)

    print(result)
