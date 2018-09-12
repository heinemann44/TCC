import numpy as np
from random import randint
import string
from PIL import Image, ImageFont, ImageDraw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tamanho_fonte', default=16, type=int)
parser.add_argument('--quantidade_imagens', default=128, type=int)
parser.add_argument('--caminho', default="./Data", type=str)


args = parser.parse_args()


class Gerador:
    lista_cacaracteres = list(string.ascii_letters) + list(string.digits)
    quantidade_letras_por_imagem = 1
    tamanho_fonte = args.tamanho_fonte
    dimensao_imagem = (28, 28)
    pasta_imagens = 'Treinar'

    def gerar_base_imagens(self, pasta = pasta_imagens):
        linha_texto_gerado = self.gerar_linhas_texto()
        fonte = ImageFont.truetype(f'{args.caminho}/Fonts/calibril.ttf', self.tamanho_fonte)
        letras = []
        for index, letra in enumerate(linha_texto_gerado):
            self.criar_imagem(letra, fonte, f'{index}.png')
            letras.append(f'{index}.png, {letra}')
        self.criar_csv(letras)

    def gerar_linhas_texto(self):
        retorno = []
        for index in range(args.quantidade_imagens):
            caracter = np.random.choice(self.lista_cacaracteres, self.quantidade_letras_por_imagem)
            retorno.append(''.join(caracter))
        return retorno

    def criar_imagem(self, letra, fonte, nome_arquivo):
        img = Image.new('RGB', self.dimensao_imagem, "black")
        draw = ImageDraw.Draw(img)
        posicao_letra = self.posicao_letra()

        draw.text(posicao_letra, letra, (255, 255, 255), font=fonte)
        img.save(f'{args.caminho}/{self.pasta_imagens}/' + nome_arquivo)

    def posicao_letra(self):
        cordenada_x = randint(30, 56)
        cordenada_y = randint(30, 52)
        posicao = ((cordenada_x - self.dimensao_imagem[0]) // 2, (cordenada_y - self.dimensao_imagem[1]) // 2)
        return posicao

    def criar_csv(self, letras):
        with open(f'{args.caminho}/{self.pasta_imagens}/Train.csv', 'w') as csv:
            csv.write('\n'.join(letras))


if __name__ == "__main__":
    gerador = Gerador()
    gerador.gerar_base_imagens()
