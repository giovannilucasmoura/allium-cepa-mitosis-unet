import PIL
import numpy as np
import matplotlib.pyplot as plt

# Funções úteis para o projeto
def redimensionar(imagem, tamanho, anotacao = False):
    imgLargura, imgAltura = imagem.size
    largura, altura = tamanho

    escala = min(largura / imgLargura, altura / imgAltura)
    novaLargura = int(imgLargura * escala)
    novaAltura = int(imgAltura * escala)

    if(anotacao):
        imagem = imagem.resize((novaLargura, novaAltura), PIL.Image.NEAREST)
        nova_imagem = PIL.Image.new('L', tamanho, (0))
    else: 
        imagem = imagem.resize((novaLargura, novaAltura), PIL.Image.BICUBIC)
        nova_imagem = PIL.Image.new('RGB', tamanho, (128, 128, 128))
    
    nova_imagem.paste(imagem, ((largura - novaLargura) // 2, (altura - novaAltura) // 2))

    return nova_imagem

def normalizar(imagem):
    imagem = imagem / 127.5 - 1
    return imagem

# Garante que a imagem tem 3 canais de cores
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else: # Se não tiver, converter
        image = image.convert('RGB')
        return image

def visualizar_imagem(imagem):
    plt.imshow(imagem)
    plt.show()