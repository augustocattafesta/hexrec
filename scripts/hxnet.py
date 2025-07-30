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
from hexrec.network import ModelBase, ModelGNN, ModelDNN, GNNRegression



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

    # Define NN model
    gnn_model = GNNRegression()
    model = ModelGNN(gnn_model)

    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'],
                              header['pitch'], kwargs['gamma'], model)

    xdata = []
    ydata = []
    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if kwargs['npixels'] == -1 or cluster.size() == kwargs['npixels']:
            xdata.append(cluster.xdata)
            mc_event = input_file.mc_event(i)
            ydata.append([mc_event.absx - cluster.x[0], mc_event.absy - cluster.y[0]])

    input_file.close()

    xdata = np.array(xdata)
    ydata = np.array(ydata) / header['pitch']

    model.train(xdata, ydata, epochs=kwargs['epochs'])
    model.save(kwargs['nnmodel'])

if __name__ == '__main__':
    hxnet(**vars(PARSER.parse_args()))
