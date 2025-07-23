"""Test for network.py
"""

import numpy as np

import keras
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

from hexrec.network import ModelBase

def test_model_base():
    inputs = Input(shape=(2,))
    h = Dense(4)(inputs)
    h = Dense(4)(h)
    outputs = Dense(1)(h)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile('adam', loss='mse')

    model_name = 'test'
    base = ModelBase(model)
    path = base.save(model_name)

    loaded = ModelBase.load(path)
    # print(loaded.model.summary())

def test_load_pretrained():
    pretrained = ModelBase.load_pretrained()
    
    xdata = np.random.randint(0, 100, size=(3, 3))
    xdata[0, 2] = 0
    xdata[2, 2] = 0
    xdata = xdata / np.sum(xdata)

    print(pretrained.evaluate(np.array([xdata])))


if __name__ == '__main__':
    test_model_base()
    test_load_pretrained()
