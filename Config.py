import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 1, 'tamanho do batch')
flags.DEFINE_integer('epoch', 2, 'epoch')
flags.DEFINE_boolean('is_training', False, 'Define se o modelo deve ser treinado')
flags.DEFINE_boolean('letra_cor_preta', False, 'Caso a letra esteja na cor branca essa flag deve ser False')
flags.DEFINE_integer('num_threads', 1, 'Número de threads para gerenciar os exemplos')
flags.DEFINE_string('logdir', 'logdir', 'Diretório de logs')
flags.DEFINE_string('results', 'results', 'caminho para armazenar os resultados')
flags.DEFINE_string('caminho_imagem_entrada', 'amostra.jpg', 'caminho onde se encontra a imagem de entrada')

config = tf.app.flags.FLAGS

letras = {0:  '0',
          1:  '1',
          2:  '2',
          3:  '3',
          4:  '4',
          5:  '5',
          6:  '6',
          7:  '7',
          8:  '8',
          9:  '9',
          10: 'a',
          11: 'b',
          12: 'c',
          13: 'd',
          14: 'e',
          15: 'f',
          16: 'g',
          17: 'h',
          18: 'i',
          19: 'j',
          20: 'k',
          21: 'l',
          22: 'm',
          23: 'n',
          24: 'o',
          25: 'p',
          26: 'q',
          27: 'r',
          28: 's',
          29: 't',
          30: 'u',
          31: 'v',
          32: 'w',
          33: 'x',
          34: 'y',
          35: 'z',
          36: 'A',
          37: 'B',
          38: 'C',
          39: 'D',
          40: 'E',
          41: 'F',
          42: 'G',
          43: 'H',
          44: 'I',
          45: 'J',
          46: 'K',
          47: 'L',
          48: 'M',
          49: 'N',
          50: 'O',
          51: 'P',
          52: 'Q',
          53: 'R',
          54: 'S',
          55: 'T',
          56: 'U',
          57: 'V',
          58: 'W',
          59: 'X',
          60: 'Y',
          61: 'Z'}
