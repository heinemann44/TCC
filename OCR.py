import tensorflow as tf
from Input import Input
from Config import config, letras
from RedeNeural import RedeNeural
from Segmentar import segmentar


class Ocr:

    def ocr(self, caminho_imagem):
        array_texto = segmentar.segmentar_imagem(caminho_imagem=caminho_imagem, inverter_imagem=config.letra_cor_preta)
        saida = ''

        for linha in array_texto:
            for palavra in linha:
                for letra in palavra:
                    predicao_letra = self._ativar_rede_neural('.Data/Letra/'+ letra)
                    saida += str(predicao_letra)
                saida += ' '
            saida += '\n'

        predicao_letra = self._retornar_letra(saida)

        self._criar_arquivo_text(predicao_letra)

    def _criar_arquivo_text(self, texto):
        arquivo = open('saida.txt', 'w+')
        arquivo.write(texto)
        arquivo.close()

    def _ativar_rede_neural(self, caminho_letra):
        iterator = Input.pegar_batch(pasta_dados=caminho_letra)

        imagens = iterator.get_next()

        cnn = RedeNeural()

        logits = cnn.construir_arquitetura(imagens)
        letra = self._retornar_letra(logits)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            saver = tf.train.Saver()
            saver.restore(sess, './Output/model.ckpt')
            saida = sess.run(letra)
            return saida

    def _decode_one_hot(self, one_hot):
        index = tf.argmax(one_hot, 1)
        return index

    def _retornar_letra(self, one_hot):
        lista_letras = letras
        index_letra = self._decode_one_hot(one_hot)
        return lista_letras.get(index_letra)


if __name__ == "__main__":
        tf.app.run()
        ocr = Ocr()
        ocr.ocr()