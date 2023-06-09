import keras

def UNet(tamanho_entrada, num_classes):
   entrada = keras.layers.Input(shape=tamanho_entrada + (3,)) # Tamanho da imagem de entrada e 3 camadas de cores RGB
   
   # qtd_filtros = [64, 128, 256, 512, 1024] # Modelo maior
   qtd_filtros = [32, 64, 128, 256, 512]  # Modelo menor

   # Contração
   caracteristicas1, contracao1 = bloco_contracao(entrada, qtd_filtros[0]) 
   caracteristicas2, contracao2 = bloco_contracao(contracao1, qtd_filtros[1])
   caracteristicas3, contracao3 = bloco_contracao(contracao2, qtd_filtros[2])
   caracteristicas4, contracao4 = bloco_contracao(contracao3, qtd_filtros[3])

   # Camadas intermediárias/continuação
   continuacao = convolucao_dupla(contracao4, qtd_filtros[4])

   # Expansão
   expansao1 = bloco_expansao(continuacao, caracteristicas4, qtd_filtros[3])
   expansao2 = bloco_expansao(expansao1, caracteristicas3, qtd_filtros[2])
   expansao3 = bloco_expansao(expansao2, caracteristicas2, qtd_filtros[1])
   expansao4 = bloco_expansao(expansao3, caracteristicas1, qtd_filtros[0])
   
   saida = keras.layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(expansao4)

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