import keras
from dataset import Dataset
from unet import UNet
import utils

# -------------------------------
# Configurações de treinamento
# -------------------------------

# Epocas de treinamento
EPOCHS = 50 

# Dimensões da imagem de entrada - altura e largura
TAMANHO_ENTRADA = (256, 192)
# TAMANHO_ENTRADA = (512, 384)
# TAMANHO_ENTRADA = (1024, 768)

# Quantidade de imagens que o modelo deve analizar por epoca de treinamento
# isso é, antes de modificar os pesos do modelo
BATCH_SIZE = 2

# Taxa de aprendizado do otimizador
LEARNING_RATE = 0.000005

dataset_treinamento = Dataset('datasets/dataset_treinamento_voc/', TAMANHO_ENTRADA, batch_size=BATCH_SIZE)
dataset_validacao = Dataset('datasets/dataset_validacao_voc/', TAMANHO_ENTRADA, batch_size=1)

unet = UNet(TAMANHO_ENTRADA, dataset_treinamento.num_classes)

unet.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy'])

historico = unet.fit(dataset_treinamento,
        validation_data=dataset_validacao,
        epochs = EPOCHS)

# Formato de nome da pasta do modelo baseado nos parametros usados
unet.save('modelos/modelo_' + str(EPOCHS) + '_' + str(TAMANHO_ENTRADA) + '_' + 
    str(BATCH_SIZE) + '_' + str(LEARNING_RATE).replace('.', ','), overwrite=True)