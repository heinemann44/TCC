import tensorflow as tf
from Input import Input
from Config import config, letras
from RedeNeural import RedeNeural
from Segmentar import Segmentar



def ocr(caminho_imagem=config.caminho_imagem_entrada):
    seg = Segmentar()
    array_texto = seg.segmentar_imagem(caminho_imagem=caminho_imagem, inverter_imagem=config.letra_cor_preta)
    texto = ''
    
    inp = Input()
    iterator = inp.pegar_batch(pasta_dados='./Data/Letra/')  
    imagens = iterator.get_next()
    print("shape img: {}".format(imagens.get_shape().as_list()))
    cnn = RedeNeural()
    logits = cnn.construir_arquitetura(imagens)
    print("shape logits: {}".format(logits.get_shape().as_list()))
    letra_predisao = tf.argmax(logits, 1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        saver.restore(sess, './Output/model.ckpt')
        for linha in array_texto:
            for palavra in linha:
                for letra in palavra:
                    saida = sess.run(letra_predisao)
                    fodase_letra = retornar_letra(saida[0])
                    print('letra: '+ str(type(fodase_letra)))
                    print('saida: '+ str(type(saida)))
                    texto += fodase_letra
                texto += ' '
            texto += '\n'
   
    _criar_arquivo_text(texto)

def _criar_arquivo_text(texto):
    arquivo = open('saida.txt', 'w+')
    arquivo.write(texto)
    arquivo.close()

def _ativar_rede_neural(caminho_letra):
    inp = Input()
    iterator = inp.pegar_batch(pasta_dados='./Data/Letra/0.jpg')
    

    imagens = iterator.get_next()
    print("shape img: {}".format(imagens.get_shape().as_list()))

    cnn = RedeNeural()

    logits = cnn.construir_arquitetura(imagens)
    print("shape logits: {}".format(logits.get_shape().as_list()))
    letra = tf.argmax(logits, 1)


    with tf.Session() as sess:
        tf.reset_default_graph()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        saver.restore(sess, './Output/model.ckpt')
        saida = sess.run(letra)
        saida = retornar_letra(saida)
        return saida

def retornar_letra(id_letra):
    return letras.get(id_letra)

def main(argv=None):
    ocr()

if __name__ == "__main__":
    tf.app.run()