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

        self.num_classes = self.num_classes(caminho)
        self.imagens, self.anotacoes = self.carregar_dados(caminho)

    # Carrega os arquivos no dataset, com imagens e anotações no formato VOC
    def carregar_dados(self, caminho):
        imagens = []
        anotacoes = []
        nomes_imagens = os.listdir(caminho + "/JPEGImages")

        for nome_imagem in nomes_imagens:
            nome_imagem = nome_imagem.split(".")[0]

            imagem = utils.processar_imagem(caminho + "/JPEGImages/" + nome_imagem + ".jpg", self.tamanho_imagem)
            anotacao = utils.processar_anotacao(caminho + "/SegmentationClassPNG/" + nome_imagem + ".png", self.tamanho_imagem)

            imagens.append(imagem)
            anotacoes.append(anotacao)

        return imagens, anotacoes
    
    # Define a quantidade de classes baseado no arquivo class_names.txt
    def num_classes(self, caminho):
        with open(caminho + 'class_names.txt') as reader:
            return len(reader.readlines())

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