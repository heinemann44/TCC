import tensorflow as tf
import numpy as np
import os
import string
from skimage.io import imread
from PIL import Image
from PIL.ExifTags import TAGS
from numpy import array



def carregarDados(caminho_dados):
    with tf.name_scope("load_data"):        
        caminho_csv = os.path.join(caminho_dados, 'Train.csv')
        imagens, texto_imagens, = [], []
        with open(caminho_csv) as arquivoTreino:
            for linha in arquivoTreino:                                                 # Passa no arquivo CSV linha por linha
                arquivo, texto = linha.strip().split(',')                               # Variável arquivo contem o nome do arquivo, texto contem o texto da imagem
                texto_imagens.append(_converterPara1Hot(texto))                     # texto_imagens contem um array 1 hot correspondente ao texto de cada imagem
                imagens.append(imread(os.path.join(caminho_dados, arquivo)))        # imagens contem um array com todas as imagens
        retorno_imagens = np.array(imagens)

        return tf.image.rgb_to_grayscale(retorno_imagens), texto_imagens


def _converterPara1Hot(entrada):
    data = list(string.digits) + list(string.ascii_letters)
    values = array(data)
    for i in range(len(values)):
        if entrada == values[i]:
            label = i
            labels = tf.one_hot(label, depth=len(data), dtype=tf.float32)
            return labels

def _extrair_metadaddos(caminho_imagem):
    imgFile = Image.open(caminho_imagem)
    info = imgFile._getexif()
    if info:
        return info


def conferir_metadados(caminho_imagem):
    metadados = _extrair_metadaddos(caminho_imagem)
    for (tag, value) in metadados.items():
        tagname = TAGS.get(tag, tag)
        if (tagname == "Model") and (value == "Letra"):
            return True
        else:
            return False


def pegar_batch(tamanho_batch, pasta_dados):
        dados, labels = carregarDados(caminho_dados=pasta_dados)
        dados = tf.cast(dados, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dados = tf.data.Dataset.from_tensor_slices(dados)
        labels = tf.data.Dataset.from_tensor_slices(labels)

        train_dataset = tf.data.Dataset.zip((dados, labels)).shuffle(5000).repeat().batch(tamanho_batch)

        return train_dataset.make_initializable_iterator()