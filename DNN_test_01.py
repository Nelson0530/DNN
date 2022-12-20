import tensorflow as tf
from keras import Sequential, layers

x = tf.random.normal([3, 784])

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu)
])

out = model(x)
print(out.shape)
