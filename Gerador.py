import numpy as np
import string
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
        linha_texto_gerado = self.gerar_linhas_texto()
        fonte = ImageFont.truetype(f'{argumentos.caminho}/Fonts/calibril.ttf', self.tamanho_fonte)
        letras = []
        for index, letra in enumerate(linha_texto_gerado):
            self.criar_imagem(letra, fonte, f'{index}.png')
            letras.append(f'{index}.png, {letra}')
        self.criar_csv(letras)

    def gerar_linhas_texto(self):
        retorno = []
        for index in range(argumentos.quantidade_imagens):
            caracter = np.random.choice(self.lista_cacaracteres, self.quantidade_letras_por_imagem)
            retorno.append(''.join(caracter))
        return retorno

    def criar_imagem(self, letra, fonte, nome_arquivo):
        img = Image.new('RGB', self.dimensao_imagem, "black")
        draw = ImageDraw.Draw(img)
        posicao_letra = self.posicao_letra()

        draw.text(posicao_letra, letra, (255, 255, 255), font=fonte)
        img.save(f'{argumentos.caminho}/{self.pasta_imagens}/' + nome_arquivo)

    def posicao_letra(self):
        cordenada_x = randint(30, 56)
        cordenada_y = randint(30, 52)
        posicao = ((cordenada_x - self.dimensao_imagem[0]) // 2, (cordenada_y - self.dimensao_imagem[1]) // 2)
        return posicao

    def criar_csv(self, letras):
        with open(f'{argumentos.caminho}/{self.pasta_imagens}/Train.csv', 'w') as csv:
            csv.write('\n'.join(letras))


if __name__ == "__main__":
    gerador = Gerador()
    gerador.gerar_base_imagens()
