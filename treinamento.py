import keras
from dataset import Dataset
from unet import UNet, TAMANHO_ENTRADA

dataset_treinamento = Dataset('dataset_teste_voc/train', (TAMANHO_ENTRADA, TAMANHO_ENTRADA))
dataset_validacao = Dataset('dataset_teste_voc/val', (TAMANHO_ENTRADA, TAMANHO_ENTRADA))

unet = UNet()
unet.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.00001),
    loss ='categorical_crossentropy',
    metrics = ['accuracy'])

unet.fit(dataset_treinamento,
        epochs = 10,
        validation_data=dataset_validacao)