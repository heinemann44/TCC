from localization import localization
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import cv2 as cv


class Segmentar:

    def _converter_para_binario(self, caminho_imagem):
        imagem = imread("0.png", as_gray=True)
        imagem_cinza = imagem * 255
        valor_limite = threshold_otsu(imagem_cinza)
        imagem_binario = imagem_cinza > valor_limite
        return imagem_binario

    def _adicionar_fundo(self, imagem):
        fundo = imread("fundo.png")
        condenada_x = 10
        condenada_y = 10
        fundo[condenada_y:condenada_y + imagem.shape[0], condenada_x:condenada_x + imagem.shape[1]] = imagem
        return fundo

    def extrair_letra(self, caminho_imagem = "0.png", caminho_salvar = "./Letras/"):

        imagem = imread(caminho_imagem)
        imagem_binaria = self._converter_para_binario(caminho_imagem)
        regioes_imagem = measure.label(imagem_binaria)

        nome_imagem = 0
        for regiao in regionprops(regioes_imagem):
            nome_imagem += 1
            if regiao.area > 3:
                minRow, minCol, maxRow, maxCol = regiao.bbox
                letra = imagem[minRow:maxRow, minCol: maxCol]

                letra_com_fundo = self._adicionar_fundo(letra)
                cv.imwrite(caminho_salvar + str(nome_imagem) + ".png", letra_com_fundo)

