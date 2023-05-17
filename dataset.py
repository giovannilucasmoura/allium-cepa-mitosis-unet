import PIL
import keras
import os
import numpy as np
import utils
import math
# Dataset carregado de uma pasta com imagens e anotações no formato VOC

# Herda da classe sequence que serve para organizar sequências de dados para o keras
# Obrigatoriamente tem que conter os metodos __getitem__ e __len__
class Dataset(keras.utils.Sequence):
    def __init__(self, caminho, tamanho_imagem, batch_size = 1):
        self.tamanho_imagem = tamanho_imagem
        self.batch_size = batch_size

        self.imagens, self.anotacoes = self.carregarDados(caminho)

    # Carrega os arquivos no dataset, com imagens e anotações no formato VOC
    def carregarDados(self, caminho):
        imagens = []
        anotacoes = []
        nomes_imagens = os.listdir(caminho + "/JPEGImages")

        for nome_imagem in nomes_imagens:
            nome_imagem = nome_imagem.split(".")[0]

            imagem = PIL.Image.open(caminho + "/JPEGImages/" + nome_imagem + ".jpg")
            imagem = utils.cvtColor(imagem)
            
            anotacao = PIL.Image.open(caminho + "/SegmentationClassPNG/" + nome_imagem + ".png")
            anotacao = PIL.Image.fromarray(np.array(anotacao))

            # Redimensionamento
            imagem = utils.redimensionar(imagem, self.tamanho_imagem, False)
            anotacao = utils.redimensionar(anotacao, self.tamanho_imagem, True)

            # Convertendo para numpy
            imagem = np.array(imagem, np.float32)
            anotacao = np.array(imagem)

            # Normalizacao da imagem
            imagem = utils.normalizar(imagem)

            imagens.append(imagem)
            anotacoes.append(anotacao)

        return imagens, anotacoes
    
    def __getitem__(self, index):
        imagens = []
        anotacoes = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            imagens.append(self.imagens[i])
            anotacoes.append(self.anotacoes[i])

        imagens = np.array(imagens)
        anotacoes = np.array(anotacoes)

        return imagens, anotacoes

    def __len__(self):
        return math.ceil(len(self.imagens) / self.batch_size)


if __name__ == "__main__":
    dataset = Dataset('dataset_teste_voc/train', (512, 512))
