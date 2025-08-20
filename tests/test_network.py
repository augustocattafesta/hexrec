"""Test for network.py
"""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from keras.layers import Input, Dense
from keras.models import Model

from hexrec.network import ModelDNN, ModelGNN


def test_save_load_DNN(size=1000):
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

    modelbase = ModelDNN(model)
    logger.info('Training the network')
    history = modelbase.train(xdata_train, ydata_train, 30, verbose=False)
    logger.info('Training completed')

    model_name = 'test'
    model_path = modelbase.save(model_name)
    logger.info(f'NN Model save to {model_path}')

    # Loading the model
    loaded_model = ModelDNN.load(model_path)
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

def test_load_pretrained_DNN(size=10):
    """Test of pretrained models
    """
    pretrained = ModelDNN.load_pretrained()
    logger.info(f'{pretrained.model.summary()}')

    # Test with some data
    pha = np.random.randint(0, 100, 7)
    xpos = np.array([ 0.,  -0.5,  0.5,  1.,   0.5, -1.,  -0.5])
    ypos = np.array([ 0., -0.8660254,  0.8660254,  0., -0.8660254,  0., 0.8660254])
    x_data = np.array([pha/pha.sum(), xpos, ypos]).T.flatten()
    predictions = pretrained.predict(np.array([x_data]))
    x_pred = predictions[:, 0]
    y_pred = predictions[:, 1]

    logger.info(f'Pretrained network prediction: {x_pred, y_pred}')

def test_event_to_graph():
    pha = np.random.randint(0, 100, 7)
    xpos = np.array([ 0.,  -0.5,  0.5,  1.,   0.5, -1.,  -0.5])
    ypos = np.array([ 0., -0.8660254,  0.8660254,  0., -0.8660254,  0., 0.8660254])
    xdata = np.array([pha, xpos, ypos]).T
    ydata = np.array([0.1, 0.1])

    logger.info(f'Array shape: {xdata.shape}')

    graph = ModelGNN.event_to_graph(xdata)
    logger.info(f'Graph without ydata: {graph}')
    graph_xy = ModelGNN.event_to_graph(xdata, ydata)
    logger.info(f'Graph with ydata: {graph_xy}')

    data_loaded = ModelGNN.data_loader(xdata)
    logger.info(f'Data loaded: {data_loaded}')
    data_loaded_xy = ModelGNN.data_loader(xdata, ydata)
    logger.info(f'Data loaded with ydata: {data_loaded_xy}')

def test_load_GNN():
    # model = GNNRegression()
    # instance = ModelGNN(model)
    # instance.load('/home/augusto/hexrec/hexrec/models/modelGNN.pt')

    model = ModelGNN.load_pretrained()
    

if __name__ == '__main__':
    test_save_load_DNN()
    test_load_pretrained_DNN()
    test_load_GNN()
    test_event_to_graph()