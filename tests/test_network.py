"""Test for network.py
"""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

from hexrec.network import ModelBase

def test_save_load(size=1000):
    """Test of save, train and load methods of ModelBase class
    """
    xdata_train = np.random.uniform(0, 1, size=(size, 1))
    ydata_train = xdata_train**2 + 1

    # Create a simple NN
    inputs = Input(shape=(1,))
    h = Dense(64, activation='relu')(inputs)
    h = Dense(64, activation='relu')(h)
    outputs = Dense(1)(h)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile('adam', loss='mse')

    modelbase = ModelBase(model)
    logger.info('Training the network')
    history = modelbase.train(xdata_train, ydata_train, 30, verbose=False)
    logger.info('Training completed')

    model_name = 'test'
    model_path = modelbase.save(model_name)
    logger.info(f'NN Model save to {model_path}')

    # Loading the model
    loaded_model = ModelBase.load(model_path)
    logger.info(f'NN Model loaded from {model_path}')
    logger.info(f'{loaded_model.model.summary()}')

    xdata_test = np.random.uniform(0, 1, size=(int(size/2), 1))
    ydata_test = loaded_model.predict(xdata_test)

    plt.figure('Data')
    plt.scatter(xdata_train, ydata_train, s=0.1, label='Train data')
    plt.scatter(xdata_test, ydata_test, s=0.1, label='Test data')
    plt.legend()

    plt.figure('History')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')

def test_load_pretrained(size=10):
    """Test of pretrained models
    """
    pretrained = ModelBase.load_pretrained()
    logger.info(f'{pretrained.model.summary()}')

    # Test with some data
    pha = np.random.randint(0, 100, 7)
    xpos = np.array([ 0.,  -0.5,  0.5,  1.,   0.5, -1.,  -0.5])
    ypos = np.array([ 0., -0.8660254,  0.8660254,  0., -0.8660254,  0., 0.8660254])
    x_data = np.array([pha/pha.sum(), xpos, ypos]).T.flatten()
    y_pred = pretrained.predict(np.array([x_data]))

    logger.info(f'Pretrained network prediction: {y_pred[0]}')

if __name__ == '__main__':
    test_save_load()
    test_load_pretrained()
