import tensorflow as tf
import numpy as np
import os
from skimage.io import imread
from skimage import transform
from numpy import array
import string


with tf.name_scope("load_data"):
    def carregarDados(caminhoDados = './Data/Testar'):
        caminhoCSV = os.path.join(caminhoDados, 'Train.csv')
        imagens, textoImagens, = [], []
        with open(caminhoCSV) as arquivoTreino:
            for linha in arquivoTreino:                                                 #Passa no arquivo CSV linha por linha
                arquivo, texto = linha.strip().split(',')                               #Variável arquivo contem o nome do arquivo, texto contem o texto da imagem
                textoImagens.append(texto)                                              #textoImagens contem um array com o texto de cada imagem
                imagens.append(imread(os.path.join(caminhoDados, arquivo)))
        list_images32 = [transform.resize(image,(28,28)).astype(np.float32).tolist() for image in imagens]
        #list_images32 = tf.cast(list_images32,tf.float32)
        retornoImagens = np.array(list_images32)
        array1hot = []        
        for i in range(len(textoImagens)):
            array1hot.append(converterPara1Hot(textoImagens[i]))
        return tf.image.rgb_to_grayscale(retornoImagens), array1hot

def converterPara1Hot(entrada):
    data = list(string.digits) + list(string.ascii_letters)
    values = array(data)
    for i in range(len(values)):
        if entrada == values[i]:
            label = i
            
    labels = tf.one_hot(label, depth=len(data), dtype=tf.float32)
    return labels

def pegarBatch(tamanhoBatch):
        dados, labels = carregarDados()
        dados = tf.cast(dados, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dados = tf.data.Dataset.from_tensor_slices(dados)
        labels = tf.data.Dataset.from_tensor_slices(labels)

        
        train_dataset = tf.data.Dataset.zip((dados, labels)).shuffle(5000).repeat().batch(tamanhoBatch)
       

        return train_dataset.make_initializable_iterator()