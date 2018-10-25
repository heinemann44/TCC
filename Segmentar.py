import cv2 as cv
import numpy as np
from ww import f
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops


class Segmentar:

    def _converter_para_binario(self, imagem):
        imagem_cinza = imagem * 255
        valor_limite = threshold_otsu(imagem_cinza)
        imagem_binario = imagem_cinza > valor_limite
        return imagem_binario

    def _inverter_imagem_binario(self, caminho_imagem):
        return np.invert(caminho_imagem)

    def _gerar_lista_letras(self, imagem_binaria):
        cordenadas_letras = []
        regioes_imagem = measure.label(imagem_binaria)
        for region in regionprops(regioes_imagem):
            if region.area < 5:
                #ignorar regiões pequenas como pontos
                continue
            cordenadas_letras.append(region.bbox)
        return cordenadas_letras

    def _separar_linhas(self, lista_letras):
        linha = []
        retorno = []
        altura_linha = lista_letras[0][0]
        for letra in lista_letras:
            diferenca_altura = abs(altura_linha - letra[0])
            if diferenca_altura < 9:
                linha.append(letra)
            else:
                altura_linha = letra[0]
                retorno.append(linha)
                linha = []
                linha.append(letra)
        retorno.append(linha)
        return retorno

    def _ordenar_linhas(self, texto_array):
        texto_ordenado = []
        for linha in texto_array:
            texto_ordenado.append(sorted(linha, key=lambda tup: tup[1]))
        return texto_ordenado

    def _separar_palavras(self, array_texto):
        texto_completo = []
        palavras_linha = []
        palavra = []
        index_linha = 0

        for linhas in array_texto:
            letra_auxiliar = array_texto[index_linha][0]
            direita_letra = letra_auxiliar[3]
            index_linha += 1
            for letra in linhas:
                if direita_letra != letra[3]:
                    proximidade = abs(direita_letra - letra[1])
                    if proximidade < 4:
                        palavra.append(letra)
                        direita_letra = letra[3]
                    else:
                        palavras_linha.append(palavra)
                        direita_letra = letra[3]
                        palavra = []
                        palavra.append(letra)
                else:
                    palavra.append(letra_auxiliar)

            palavras_linha.append(palavra)
            texto_completo.append(palavras_linha)
            palavras_linha = []
            palavra = []

        return texto_completo

    def _salvar_imagem_texto(self, texto_array, caminho_imagem):
        imagem = imread(caminho_imagem)
        nome_imagem = 0
        texto = []
        for linha in texto_array:
            linhas = []
            texto.append(linhas)
            for palavra in linha:
                palavras = []
                linhas.append(palavras)
                for letra in palavra:
                    palavras.append(nome_imagem)

                    aresta_topo = letra[0]
                    aresta_esquerda = letra[1]
                    aresta_base = letra[2]
                    aresta_direita = letra[3]

                    letra_sem_fundo = imagem[aresta_topo:aresta_base, aresta_esquerda: aresta_direita]
                    self._salvar_letra(nome="letra", imagem=letra_sem_fundo, caminho_salvar="")
                    letra_sem_fundo2 = imread("letra.jpg")
                    letra_com_fundo = self._adicionar_fundo(letra_sem_fundo2)
                    self._salvar_letra(nome=nome_imagem, imagem=letra_com_fundo)

                    nome_imagem += 1

        return texto

    def _adicionar_fundo(self, imagem):
        fundo = imread("fundo.jpg")
        condenada_x = 10
        condenada_y = 10
        fundo[condenada_y:condenada_y + imagem.shape[0], condenada_x:condenada_x + imagem.shape[1]] = imagem
        return fundo

    def _salvar_letra(self, nome, imagem, caminho_salvar="./Data/Letra/"):
        cv.imwrite(f('{caminho_salvar}{str(nome)}.jpg'), imagem)

    def segmentar_imagem(self, caminho_imagem="amostra.jpg", inverter_imagem=True):

        imagem = imread(caminho_imagem, as_gray=True)
        imagem = self._converter_para_binario(imagem)
        if inverter_imagem:
            imagem = self._inverter_imagem_binario(imagem)

        texto_array = self._gerar_lista_letras(imagem)
        texto_array = self._separar_linhas(texto_array)
        texto_array = self._ordenar_linhas(texto_array)
        texto_array = self._separar_palavras(texto_array)

        texto = self._salvar_imagem_texto(texto_array=texto_array, caminho_imagem=caminho_imagem)
        return texto


if __name__ == "__main__":

    segmentar = Segmentar()
