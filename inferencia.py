import keras
import tensorflow as tf
import os
import utils
import numpy as np
import PIL
# ----------------
# Visualização de inferência e avaliação de modelo treinado
# ----------------

# Caminho da pasta onde o modelo está salvo, no formato SavedModel
caminho_modelo = 'modelos/modelo_50_(256, 192)_2_5e-06'
caminho_dados = 'datasets/dataset_avaliacao_voc/'
mostrar_camada_resposta = False

# Carregando modelo e tamanho do formato da camada de entrada
modelo = keras.models.load_model(caminho_modelo)
camada_entrada = modelo.layers[0]
tamanho_entrada = (camada_entrada.output_shape[0][2], camada_entrada.output_shape[0][1])

# Se os dados apontarem diretamente para uma imagem, fazer a predição
if(os.path.isfile(caminho_dados)):
    imagem = PIL.Image.open(caminho_dados)
    imagem_predicao = utils.processar_imagem(caminho_dados, tamanho_entrada)
    imagem_predicao = np.expand_dims(imagem_predicao, 0)
    predicao = modelo.predict(imagem_predicao)
    
    if mostrar_camada_resposta:
        predicao = tf.argmax(predicao, axis=-1)
        predicao = predicao[..., tf.newaxis]
        predicao = tf.keras.preprocessing.image.array_to_img(predicao[0])
    else:
        predicao = tf.keras.preprocessing.image.array_to_img(np.expand_dims(predicao[0][:,:,1], axis=-1))
    
    predicao = utils.redimensionar_anotacao(predicao, imagem.size)
    utils.visualizar(imagem, [], predicao)

    

# Se os dados forem uma pasta, assumir que é o formato VOC e também buscar/mostrar as anotações
if(os.path.isdir(caminho_dados)):
    nomes_imagens = os.listdir(caminho_dados + "/JPEGImages")
    imagens = []
    imagens_predicao = []
    anotacoes = []

    for nome_imagem in nomes_imagens:
        nome_imagem = nome_imagem.split(".")[0]

        imagem = utils.processar_imagem(caminho_dados + "/JPEGImages/" + nome_imagem + ".jpg", None)
        imagem_predicao = utils.processar_imagem(caminho_dados + "/JPEGImages/" + nome_imagem + ".jpg", tamanho_entrada)
        imagem_predicao = np.expand_dims(imagem_predicao, 0)
        anotacao = utils.processar_anotacao(caminho_dados + "/SegmentationClassPNG/" + nome_imagem + ".png", None)

        imagens.append(imagem)
        imagens_predicao.append(imagem_predicao)
        anotacoes.append(anotacao)
        
    predicoes = modelo.predict(np.vstack(imagens_predicao))

    for i in range(len(imagens)):
        predicao = predicoes[i]

        if mostrar_camada_resposta:
            predicao = tf.argmax(predicao, axis=-1)
            predicao = predicao[..., tf.newaxis]
            predicao = tf.keras.preprocessing.image.array_to_img(predicao)
        else:
            predicao = tf.keras.preprocessing.image.array_to_img(np.expand_dims(predicao[:,:,1], axis=-1))

        predicao = utils.redimensionar_anotacao(predicao, tf.keras.preprocessing.image.array_to_img(imagens[i]).size)
        utils.visualizar(imagens[i], anotacoes[i], predicao)