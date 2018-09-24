import tensorflow as tf
import numpy as np
import os
import string
from skimage.io import imread
from skimage import transform
from numpy import array


with tf.name_scope("load_data"):
    def carregarDados(caminho_dados):
        caminho_csv = os.path.join(caminho_dados, 'Train.csv')
        imagens, texto_imagens, = [], []
        with open(caminho_csv) as arquivoTreino:
            for linha in arquivoTreino:                                                 # Passa no arquivo CSV linha por linha
                arquivo, texto = linha.strip().split(',')                               # Variável arquivo contem o nome do arquivo, texto contem o texto da imagem
                texto_imagens.append(converterPara1Hot(texto))                          # texto_imagens contem um array 1 hot correspondente ao texto de cada imagem
                imagens.append(imread(os.path.join(caminho_dados, arquivo)))            # imagens contem um array com todas as imagens
        # list_images32 = [transform.resize(image, (28, 28)).astype(np.float32).tolist() for image in imagens]
        # list_images32 = tf.cast(list_images32,tf.float32)
        # retorno_imagens = np.array(list_images32)
        retorno_imagens = np.array(imagens)

        return tf.image.rgb_to_grayscale(retorno_imagens), texto_imagens


def converterPara1Hot(entrada):
    data = list(string.digits) + list(string.ascii_letters)
    values = array(data)
    for i in range(len(values)):
        if entrada == values[i]:
            label = i

    labels = tf.one_hot(label, depth=len(data), dtype=tf.float32)
    return labels


def pegarBatch(tamanho_batch, pasta_dados):
        dados, labels = carregarDados(caminho_dados=pasta_dados)
        dados = tf.cast(dados, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dados = tf.data.Dataset.from_tensor_slices(dados)
        labels = tf.data.Dataset.from_tensor_slices(labels)

        train_dataset = tf.data.Dataset.zip((dados, labels)).shuffle(5000).repeat().batch(tamanho_batch)

        return train_dataset.make_initializable_iterator()