import os
import tensorflow as tf
from Config import config
from Input import pegarBatch
from RedeConvolucional import RedeConvolucional

class RedeNeural():

    def __init__(self):
        self._definir_hyperparametros()

    def _definir_hyperparametros(self):
        self.num_canais = 1
        self.num_caracteristicas = 28
        self.num_classes = 62

    with tf.name_scope("conv_net"):
        def construir_arquitetura(self, imagens):
            with tf.name_scope("conv_net_conv1"):
                peso1 = self._criar_variavel("w0", [5, 5, 1, 256])
                bias1 = self._criar_variavel("b0", [256])
                conv1 = RedeConvolucional(peso1, bias1)
                saida1 = conv1.conv2d(imagens)
                saida1 = conv1.maxpool2d(saida1, k=2)

            with tf.name_scope("conv_net_conv2"):
                peso2 = self._criar_variavel("w1", [5, 5, 256, 256])
                bias2 = self._criar_variavel("b1", [256])
                conv2 = RedeConvolucional(peso2, bias2)
                saida2 = conv2.conv2d(saida1)
                saida2 = conv2.maxpool2d(saida2, k=2)

            with tf.name_scope("conv_net_conv3"):
                peso3 = self._criar_variavel("w2", [5, 5, 256, 128])
                bias3 = self._criar_variavel("b2", [128])
                conv3 = RedeConvolucional(peso3, bias3)
                saida3 = conv3.conv2d(saida2)
                saida3 = conv3.maxpool2d(saida3, k=2)

            with tf.name_scope("conv_net_fc1"):
                peso4 = self._criar_variavel("w3", [4 * 4 * 128, 328])
                bias4 = self._criar_variavel("b3", [328])
                fc1 = tf.contrib.layers.flatten(saida3)
                fc1 = tf.add(tf.matmul(fc1, peso4), bias4)
                fc1 = tf.nn.relu(fc1)

            with tf.name_scope("conv_net_fc2"):
                peso5 = self._criar_variavel("w4", [328, 192])
                bias5 = self._criar_variavel("b4", [192])
                fc2 = tf.add(tf.matmul(fc1, peso5), bias5)
                fc2 = tf.nn.relu(fc2)

            with tf.name_scope("conv_net_out"):
                peso6 = self._criar_variavel("w5", [192, self.num_classes])
                bias6 = self._criar_variavel("b5", [self.num_classes])
                saida = tf.add(tf.matmul(fc2, peso6), bias6)

            return saida

    with tf.name_scope("custo"):
        def custo(self, logits, labels):
            custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar("custo", custo)
            return custo

    with tf.name_scope("treino"):
        def treinar(self, custo):
            return tf.train.AdamOptimizer().minimize(custo)

    with tf.name_scope("accuracy"):
        def accuracy(self, logits, labels):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
            return accuracy

    def _criar_variavel(self, nome, shape):
        variavel = tf.get_variable(name=nome, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram(nome, variavel)
        return variavel

