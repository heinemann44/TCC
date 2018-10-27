import tensorflow as tf
import numpy as np
import os
import string
from Config import config
from skimage.io import imread
from PIL import Image
from PIL.ExifTags import TAGS
from numpy import array
from natsort import natsorted


class Input:


    def carregar_dados(self, caminho_dados, treinando=config.is_training):
        with tf.name_scope("load_data"):
            if treinando:
                caminho_csv = os.path.join(caminho_dados, 'Train.csv')
                imagens, texto_imagens, = [], []
                with open(caminho_csv) as arquivoTreino:
                    for linha in arquivoTreino:
                        arquivo, texto = linha.strip().split(',')
                        texto_imagens.append(self._converter_para_one_hot(texto))
                        imagens.append(imread(os.path.join(caminho_dados, arquivo)))
                retorno_imagens = np.array(imagens)

                return tf.image.rgb_to_grayscale(retorno_imagens), texto_imagens
            else:
                imagem = []    
                nome_diretorio = natsorted(os.listdir(caminho_dados))
                                
                for filename in nome_diretorio:
                    if 'jpg' in filename:
                        print(filename)
                        imagem.append(imread(caminho_dados +'/'+ filename))
                
                retorno_imagens = np.array(imagem)
                return tf.image.rgb_to_grayscale(retorno_imagens)

    def _converter_para_one_hot(self, entrada):
        data = list(string.digits) + list(string.ascii_letters)
        values = array(data)
        for i in range(len(values)):
            if entrada == values[i]:
                label = i
                labels = tf.one_hot(label, depth=len(data), dtype=tf.float32)
                return labels

    def _extrair_metadaddos(self, caminho_imagem):
        imgFile = Image.open(caminho_imagem)
        info = imgFile._getexif()
        if info:
            return info

    def conferir_metadados(self, caminho_imagem):
        metadados = self._extrair_metadaddos(caminho_imagem)
        for (tag, value) in metadados.items():
            tagname = TAGS.get(tag, tag)
            if (tagname == "Model") and (value == "Letra"):
                return True
            else:
                return False

    def pegar_batch(self, 
                    pasta_dados, 
                    tamanho_batch=config.batch_size, 
                    treinando=config.is_training):
        if treinando:
            dados, labels = self.carregar_dados(caminho_dados=pasta_dados)
            dados = tf.cast(dados, tf.float32)
            labels = tf.cast(labels, tf.float32)
            dados = tf.data.Dataset.from_tensor_slices(dados)
            labels = tf.data.Dataset.from_tensor_slices(labels)
            train_dataset = tf.data.Dataset.zip((dados, labels))
                              .shuffle(5000).repeat().batch(tamanho_batch)
        else:
            dados = self.carregar_dados(caminho_dados=pasta_dados)
            dados = tf.cast(dados, tf.float32)
            train_dataset = tf.data.Dataset.from_tensor_slices(dados).repeat().batch(1)
        
        return train_dataset.make_initializable_iterator()
