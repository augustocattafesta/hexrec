"""Test for network.py
"""

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

    loaded = ModelBase.load(model_name)
    print(loaded.model.summary())

if __name__ == '__main__':
    test_model_base()
