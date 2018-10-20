import numpy as np
import string
import piexif
from ww import f
from random import randint
from PIL import Image, ImageFont, ImageDraw
from ConfigGerador import argumentos, fontes


class Gerador:
    lista_cacaracteres = list(string.ascii_letters) + list(string.digits)
    quantidade_letras_por_imagem = 1
    tamanho_fonte = argumentos.tamanho_fonte
    dimensao_imagem = (28, 28)
    pasta_imagens = argumentos.pasta
    lista_fontes = fontes

    def _retornar_fonte_aleatoria(self):
        id_fonte = randint(1, 55)
        return self.lista_fontes.get(id_fonte)

    def gerar_base_imagens(self):
        linha_texto_gerado = self._gerar_linhas_texto()
        letras = []
        for index, letra in enumerate(linha_texto_gerado):
            nome_fonte = self._retornar_fonte_aleatoria()
            fonte = ImageFont.truetype(f('/usr/local/share/fonts/ms_fonts/{nome_fonte}'), self.tamanho_fonte)
            self._criar_imagem(letra, fonte, f('{index}.jpg'))
            letras.append(f('{index}.jpg,{letra}'))
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
        posicao_letra = self._posicao_letra(fonte, letra)

        draw.text(posicao_letra, letra, (255, 255, 255), font=fonte)
        metadado_modelo = self._criar_metadado()
        img.save(f('{argumentos.caminho}/{self.pasta_imagens}/{nome_arquivo}'), exif=metadado_modelo)

    def _posicao_letra(self, fonte, letra):
        tamanho_letra = fonte.getsize(letra)

        try:
            menor_coordenada_x_possivel = 1
            maior_coordenada_x_possivel = ((self.dimensao_imagem[0] - tamanho_letra[0]) - 1)
            menor_coordenada_y_possivel = 1
            maior_coordenada_y_possivel = ((self.dimensao_imagem[0] - tamanho_letra[1]) - 1)

            coordenada_x = randint(menor_coordenada_x_possivel, maior_coordenada_x_possivel)
            coordenada_y = randint(menor_coordenada_y_possivel, maior_coordenada_y_possivel)
        except:
            coordenada_x = 1
            coordenada_y = 1
        posicao = (coordenada_x, coordenada_y)
        return posicao

    def _criar_csv(self, letras):
        with open(f('{argumentos.caminho}/{self.pasta_imagens}/Train.csv'), 'w') as csv:
            csv.write('\n'.join(letras))

    def _criar_metadado(self):
        metadado_modelo = {piexif.ImageIFD.Model: u"Letra"}
        exif_dict = {"0th": metadado_modelo}
        exif_bytes = piexif.dump(exif_dict)
        return exif_bytes


if __name__ == "__main__":
    gerador = Gerador()
    gerador.gerar_base_imagens()
