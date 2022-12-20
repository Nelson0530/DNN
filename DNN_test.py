import tensorflow as tf

x = tf.random.normal([3, 784])

w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

o1 = x @ w1 + b1
s1 = tf.nn.sigmoid(o1)
o2 = s1 @ w2 + b2
s2 = tf.nn.sigmoid(o2)
o3 = s2 @ w3 + b3
s3 = tf.nn.sigmoid(o3)
o4 = s3 @ w4 + b4
print(o4.shape)

