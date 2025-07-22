"""Neural network model definition and training
"""

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam

from hexsample import logger
from hexsample.readout import HexagonalReadoutCircular
from hexsample.fileio import DigiInputFileCircular, ReconOutputFile
from hexsample.hexagon import HexagonalLayout
from hexsample.recon import ReconEvent

from hexrec.app import ArgumentParser
from hexrec.clustering import ClusteringNN
from hexrec.network import ModelBase



__description__ = \
"""Define a neural network model and train it
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_infile()
PARSER.add_model_name()
PARSER.add_reconstruction_options()
PARSER.add_neural_net_options()

def hxnet(**kwargs):
    input_file_path = kwargs['infile']
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    logger.info(f'Readout chip: {readout}')

    # Devo modificare il clustering, serve che restituisca una matrice 3x3
    # Ma dipende da quanti modelli voglio usare, se tutti CNN oppure no
    # Comunque la grid 3x3 mi da invarianza sulla posizione dei pixel perchè
    # non c'è dipendenza dagli adc, mentre se uso il pha ho dipendenza dalla posizione
    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'],
                              kwargs['gamma'])

    # devo mettere un'opzione per maskare a seconda di quanti pixel voglio, forse
    # mi basta mettere se ne voglio due oppure tutti
    xdata = []
    ydata = []
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if kwargs['npixels'] == -1 or cluster.size() == kwargs['npixels']:
            xdata.append(cluster.grid)
            mc_event = input_file.mc_event(i)
            ydata.append([mc_event.absx - cluster.x[0], mc_event.absy - cluster.y[0]])

    input_file.close()

    xdata = np.array(xdata) / np.sum(xdata, axis=(1,2), keepdims=True)
    ydata = np.array(ydata) / header['pitch']

    # Opzioni per la rete: loss, optimizer (learning rate), 
    inputs = Input(shape=(3,3,1))
    h = Conv2D(4, (2,2), activation='relu')(inputs)
    h = Flatten()(h)
    h = Dense(4, activation='relu')(h)
    outputs = Dense(2, activation='linear')(h)
    optimizer = Adam(learning_rate=0.0001)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # add argument for callbacks, with early stopping, fixing the patience, and
    # ReduceLRonplateu, fixing, factor, min lr
    # also add an option for optimizer
    # would be nice to have an output logging file with all info
    modelbase = ModelBase(model)
    modelbase.train(xdata, ydata, epochs=kwargs['epochs'])
    modelbase.save(kwargs['nnmodel'])

if __name__ == '__main__':
    hxnet(**vars(PARSER.parse_args()))
