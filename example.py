import tensorflow as tf
hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
print(1+1+1)
print(23+22)
print(2+4)