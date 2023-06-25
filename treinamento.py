import keras
from dataset import Dataset
from unet import UNet
import utils
# -------------------------------
# Configurações de treinamento
# -------------------------------

# Epocas de treinamento
EPOCHS = 25

# Dimensões da imagem de entrada - altura e largura
INPUT_SIZE = (208, 160)
# INPUT_SIZE = (416, 320)
# INPUT_SIZE = (544, 416)

# Quantidade de imagens que o modelo deve analizar por epoca de treinamento
# isso é, antes de modificar os pesos do modelo
BATCH_SIZE = 1

# Taxa de aprendizado do otimizador
LEARNING_RATE = 0.000005

# Carregamento das bases de dados
dataset_treinamento = Dataset('datasets/dataset_treinamento_voc/', INPUT_SIZE, batch_size=BATCH_SIZE)
dataset_validacao = Dataset('datasets/dataset_validacao_voc/', INPUT_SIZE, batch_size=1)

# Definição e compilação do modelo
unet = UNet(INPUT_SIZE, dataset_treinamento.num_classes)
unet.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss ='sparse_categorical_crossentropy',
    metrics = ['accuracy'])

# Treinamento
unet.fit(dataset_treinamento,
        validation_data=dataset_validacao,
        epochs = EPOCHS)

# Salvando o modelo no disco local
# Formato de nome da pasta do modelo baseado nos parametros usados
unet.save('modelos/modelo_' + str(EPOCHS) + '_' + str(INPUT_SIZE) + '_' + 
    str(BATCH_SIZE), overwrite=True)