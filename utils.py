import PIL
import numpy as np
import matplotlib.pyplot as plt
import keras
# ---------------
# Funções úteis para o projeto
# ---------------

# Pre-processamento para treinamento
def processar_imagem(caminho, tamanho):
    imagem = PIL.Image.open(caminho)
    imagem = converter_cores_rgb(imagem)
    if(tamanho != None):
        imagem = redimensionar_imagem(imagem, tamanho)
    imagem = np.array(imagem, np.float32)
    imagem = normalizar(imagem)

    return imagem

# Pre-processamento para treinamento
def processar_anotacao(caminho, tamanho):
    anotacao = PIL.Image.open(caminho)
    anotacao = PIL.Image.fromarray(np.array(anotacao))
    if(tamanho != None):
        anotacao = redimensionar_anotacao(anotacao, tamanho)
    anotacao = np.array(anotacao)
    
    return anotacao

def redimensionar_imagem(imagem, tamanho):
    imgLargura, imgAltura = imagem.size
    largura, altura = tamanho

    escala = min(largura / imgLargura, altura / imgAltura)
    novaLargura = int(imgLargura * escala)
    novaAltura = int(imgAltura * escala)

    imagem = imagem.resize((novaLargura, novaAltura), PIL.Image.BICUBIC)
    nova_imagem = PIL.Image.new('RGB', tamanho, (128, 128, 128))
    
    nova_imagem.paste(imagem, ((largura - novaLargura) // 2, (altura - novaAltura) // 2))

    return nova_imagem

def redimensionar_anotacao(anotacao, tamanho):
    anoLargura, anoAltura = anotacao.size
    largura, altura = tamanho

    escala = min(largura / anoLargura, altura / anoAltura)
    novaLargura = int(anoLargura * escala)
    novaAltura = int(anoAltura * escala)

    anotacao = anotacao.resize((novaLargura, novaAltura), PIL.Image.NEAREST)
    nova_anotacao = PIL.Image.new('L', tamanho, (0))
    
    nova_anotacao.paste(anotacao, ((largura - novaLargura) // 2, (altura - novaAltura) // 2))

    return nova_anotacao

def normalizar(imagem):
    imagem = imagem / 127.5 - 1
    return imagem

# Garante que a imagem tem 3 canais de cores
def converter_cores_rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else: # Se não tiver, converter
        image = image.convert('RGB')
        return image

def iou_mascara_binaria(anotacao, predicao):
    # Area total positiva das mascaras
    anotacao_area = np.count_nonzero(anotacao == 1)
    predicao_area = np.count_nonzero(predicao == 1)

    # Area com valor positivo em ambas as mascaras
    intersecao = np.count_nonzero(np.logical_and(anotacao == 1,  predicao == 1))

    iou = intersecao / (anotacao_area + predicao_area - intersecao)

    return iou

def visualizar(imagem, anotacao, predicao):
    plt.figure(figsize=(15, 15))

    titulo = ['Imagem', 'Resposta do modelo']

    # if len(anotacao) != 0:
    titulo.insert(1, 'Anotação')

    for i in range(len(titulo)):
        plt.subplot(1, len(titulo), i + 1)
        plt.title(titulo[i])
        if(i == 0): 
            plt.imshow(imagem)
        if(i == 1): 
            plt.imshow(anotacao)
        if(i == 2): 
            plt.imshow(predicao)
        plt.axis('off')
        
    plt.show()