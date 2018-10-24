import tensorflow as tf


class RedeConvolucional:

    def __init__(self, peso, bias):
        self.peso = peso
        self.bias = bias

    def conv2d(self, inputs, strides=1):
        saida = tf.nn.conv2d(inputs, self.peso, strides=[1, strides, strides, 1], padding='SAME')
        saida = tf.nn.bias_add(saida, self.bias)
        return tf.nn.relu(saida)

    def maxpool2d(self, inputs, k=2):
        return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    def montar_camada_convolucional(self, entrada):
        conv = self.conv2d(entrada)
        saida = self.maxpool2d(conv, k=2)
        return saida