import tensorflow as tf


class RedeTotalmenteConectada:

    def __init__(self, peso, bias):
        self.peso = peso
        self.bias = bias

    def camada_densa(self, inputs):
        return tf.add(tf.matmul(inputs, self.peso), self.bias)

    def funcao_ativacao(self, camada_entrada):
        return tf.nn.relu(camada_entrada)

    def montar_camada_convolucionar(self, entrada):
        fc = self.camada_densa(entrada)
        saida = self.funcao_ativacao(fc)
        return saida