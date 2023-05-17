import keras

## Configurações
# Dimensões da imagem de entrada, altura e largura
TAMANHO_ENTRADA = 512
# Quantidade de imagens por epoca de treinamento
IMAGENS_POR_EPOCA = 1

def UNet():
   entrada = keras.layers.Input(shape=(TAMANHO_ENTRADA, TAMANHO_ENTRADA, 3)) # Tamanho da imagem de entrada e 3 camadas de cores RGB
   
   # Contração
   caracteristicas1, contracao1 = bloco_contracao(entrada, 64) # entrada / 2, 64 filtros
   caracteristicas2, contracao2 = bloco_contracao(contracao1, 128)     # entrada / 4, 128 filtros
   caracteristicas3, contracao3 = bloco_contracao(contracao2, 256)     # entrada / 8, 256 filtros
   caracteristicas4, contracao4 = bloco_contracao(contracao3, 512)     # entrada / 16, 512 filtros

   # Camadas intermediárias/continuação
   continuacao = convolucao_dupla(contracao4, 1024)

   # Expansão
   expansao1 = bloco_expansao(continuacao, caracteristicas4, 512)
   expansao2 = bloco_expansao(expansao1, caracteristicas3, 256)
   expansao3 = bloco_expansao(expansao2, caracteristicas2, 128)
   expansao4 = bloco_expansao(expansao3, caracteristicas1, 64)
   
   saida = keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(expansao4)

   # Modelo completo
   unet_model = keras.Model(entrada, saida, name="UNet")

   return unet_model

# Camadas de convolução usados na contração
def convolucao_dupla(camadas, filtros):
   # Conv2D then ReLU activation
   camadas = keras.layers.Conv2D(filtros, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(camadas)
   # Conv2D then ReLU activation
   camadas = keras.layers.Conv2D(filtros, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(camadas)
   return camadas

# Blocos de convolução completos, com as camadas, pooling e dropout
def bloco_contracao(camadas, filtros):
   camadas_caracteristicas = convolucao_dupla(camadas, filtros)
   camadas = keras.layers.MaxPool2D(2)(camadas_caracteristicas)   # MaxPool reduz o tamanho pela metade
   camadas = keras.layers.Dropout(0.3)(camadas)   # Dropout para evitar overfit
   return camadas_caracteristicas, camadas

# Camadas usadas na expansão da imagem
def bloco_expansao(camadas, caracteristicas_anteriores, filtros):
   camadas = keras.layers.Conv2DTranspose(filtros, 3, 2, padding="same")(camadas)
   camadas = keras.layers.concatenate([camadas, caracteristicas_anteriores])
   camadas = keras.layers.Dropout(0.3)(camadas)
   camadas = convolucao_dupla(camadas, filtros)
   
   return camadas

if(__name__ == "__main__"):
    unet = unet()
    unet.summary()