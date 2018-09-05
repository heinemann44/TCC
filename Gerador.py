import numpy as np
from random import randint
import string
from PIL import Image, ImageFont, ImageDraw


def MakeImg(t, f, fn, s = (100, 100), o = (16, 8)):
    '''
    Generate an image of text
    t:      The text to display in the image
    f:      The font to use
    fn:     The file name
    s:      The image size
    o:      The offest of the text in the image
    '''
    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(OFS, t, (255, 255, 255), font = f)
    img.save('./Data/Testar/' + fn)                      # caminho que vai ser gerado a imagem

#The possible characters to use
CS = list(string.ascii_letters) + list(string.digits)
#The random strings
S = [''.join(np.random.choice(CS, 1)) for i in range(128)] # quantidade de imagens gerada
#Get the font
font = ImageFont.truetype("./Fonts/calibril.ttf", 16)
#The largest size needed
MS = max(font.getsize(Si) for Si in S)
#Image size
MS = (28, 28)
Y = []

for i, Si in enumerate(S):
    x = randint(30, 56)
    y = randint(30, 52)
    OFS = ((x - MS[0]) // 2, (y - MS[1]) // 2)
    MakeImg(Si, font, str(i) + '.png', MS, OFS)
    Y.append(str(i) + '.png,' + Si)
#Write CSV file
with open('./Data/Testar/Train.csv', 'w') as F:       # caminho que vai ser gerado o csv
    F.write('\n'.join(Y))
