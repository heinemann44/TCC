import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tamanho_fonte', default=16, type=int)
parser.add_argument('--quantidade_imagens', default=128, type=int)
parser.add_argument('--caminho', default="./Data", type=str)
parser.add_argument('--pasta', default="./Treinar", type=str)

argumentos = parser.parse_args()