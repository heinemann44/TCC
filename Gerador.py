import numpy as np
import string
import piexif
from random import randint
from PIL import Image, ImageFont, ImageDraw
from ConfigGerador import argumentos


class Gerador:
    lista_cacaracteres = list(string.ascii_letters) + list(string.digits)
    quantidade_letras_por_imagem = 1
    tamanho_fonte = argumentos.tamanho_fonte
    dimensao_imagem = (28, 28)
    pasta_imagens = argumentos.pasta

    def gerar_base_imagens(self):
        linha_texto_gerado = self._gerar_linhas_texto()
        fonte = ImageFont.truetype(f"{argumentos.caminho}/Fonts/calibril.ttf", self.tamanho_fonte)
        letras = []
        for index, letra in enumerate(linha_texto_gerado):
            self._criar_imagem(letra, fonte, f'{index}.jpg')
            letras.append(f'{index}.jpg, {letra}')
        self._criar_csv(letras)

    def _gerar_linhas_texto(self):
        retorno = []
        for index in range(argumentos.quantidade_imagens):
            caracter = np.random.choice(self.lista_cacaracteres, self.quantidade_letras_por_imagem)
            retorno.append(''.join(caracter))
        return retorno

    def _criar_imagem(self, letra, fonte, nome_arquivo):
        img = Image.new('RGB', self.dimensao_imagem, "black")
        draw = ImageDraw.Draw(img)
        posicao_letra = self._posicao_letra()

        draw.text(posicao_letra, letra, (255, 255, 255), font=fonte)
        metadado_modelo = self._criar_metadado()
        img.save(f'{argumentos.caminho}/{self.pasta_imagens}/' + nome_arquivo, exif=metadado_modelo)

    def _posicao_letra(self):
        cordenada_x = randint(30, 56)
        cordenada_y = randint(30, 52)
        posicao = ((cordenada_x - self.dimensao_imagem[0]) // 2, (cordenada_y - self.dimensao_imagem[1]) // 2)
        return posicao

    def _criar_csv(self, letras):
        with open(f"{argumentos.caminho}/{self.pasta_imagens}/Train.csv", 'w') as csv:
            csv.write('\n'.join(letras))

    def _criar_metadado(self):
        metadado_modelo = {piexif.ImageIFD.Model: u"Letra"}
        exif_dict = {"0th": metadado_modelo}
        exif_bytes = piexif.dump(exif_dict)
        return exif_bytes


if __name__ == "__main__":
    gerador = Gerador()
    gerador.gerar_base_imagens()
