import tensorflow as tf

class RedeConvolucional:

    def __init__(self, peso, bias):
        self.peso = peso
        self.bias = bias

    def conv2d(self, inputs, strides=1):
        x = tf.nn.conv2d(inputs, self.peso, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.bias)
        return tf.nn.relu(x)

    def maxpool2d(self, inputs, k=2):
        return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

