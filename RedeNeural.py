import tensorflow as tf
from RedeConvolucional import RedeConvolucional
from RedeTotalmenteConectada import RedeTotalmenteConectada

class RedeNeural:

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
                conv1 = conv1.montar_camada_convolucionar(entrada=imagens)

            with tf.name_scope("conv_net_conv2"):
                peso2 = self._criar_variavel("w1", [5, 5, 256, 256])
                bias2 = self._criar_variavel("b1", [256])
                conv2 = RedeConvolucional(peso2, bias2)
                conv2 = conv2.montar_camada_convolucionar(entrada=conv1)

            with tf.name_scope("conv_net_conv3"):
                peso3 = self._criar_variavel("w2", [5, 5, 256, 128])
                bias3 = self._criar_variavel("b2", [128])
                conv3 = RedeConvolucional(peso3, bias3)
                conv3 = conv3.montar_camada_convolucionar(entrada=conv2)

            with tf.name_scope("conv_net_fc1"):
                peso4 = self._criar_variavel("w3", [4 * 4 * 128, 328])
                bias4 = self._criar_variavel("b3", [328])
                convolucional_flatten = tf.contrib.layers.flatten(conv3)
                fc1 = RedeTotalmenteConectada(peso4, bias4)
                fc1 = fc1.montar_camada_convolucionar(convolucional_flatten)

            with tf.name_scope("conv_net_fc2"):
                peso5 = self._criar_variavel("w4", [328, 192])
                bias5 = self._criar_variavel("b4", [192])
                fc2 = RedeTotalmenteConectada(peso5, bias5)
                fc2 = fc1.montar_camada_convolucionar(fc2)

            with tf.name_scope("conv_net_out"):
                peso6 = self._criar_variavel("w5", [192, self.num_classes])
                bias6 = self._criar_variavel("b5", [self.num_classes])
                fc3 = RedeTotalmenteConectada(peso6, bias6)
                saida = fc3.camada_densa(fc2)

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

