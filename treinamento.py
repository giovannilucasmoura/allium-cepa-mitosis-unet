import keras
from dataset import Dataset
from unet import UNet

# -------------------------------
# Configurações de treinamento
# -------------------------------

# Epocas de treinamento
EPOCHS = 10 

# Dimensões da imagem de entrada - altura e largura
TAMANHO_ENTRADA = 256

# Quantidade de imagens que o modelo deve analizar por epoca de treinamento
# isso é, antes de modificar os pesos do modelo
BATCH_SIZE = 2

# Taxa de aprendizado do otimizador
LEARNING_RATE = 0.00001

dataset_treinamento = Dataset('dataset_teste_voc/train/', (TAMANHO_ENTRADA, TAMANHO_ENTRADA), batch_size=BATCH_SIZE)
dataset_validacao = Dataset('dataset_teste_voc/val/', (TAMANHO_ENTRADA, TAMANHO_ENTRADA), batch_size=BATCH_SIZE)

unet = UNet(TAMANHO_ENTRADA, dataset_treinamento.num_classes)

unet.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy'])

historico = unet.fit(dataset_treinamento,
        validation_data=dataset_validacao,
        epochs = EPOCHS)