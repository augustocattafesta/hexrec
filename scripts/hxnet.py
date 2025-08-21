"""Neural network model definition and training
"""

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

from hexsample import logger
from hexsample.readout import HexagonalReadoutCircular
from hexsample.fileio import DigiInputFileCircular
from hexsample.hexagon import HexagonalLayout

from hexrec.app import ArgumentParser
from hexrec.clustering import ClusteringNN
from hexrec.network import ModelGNN, ModelDNN, GNNRegression

__description__ = \
"""Define a neural network model and train it
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_infile()
PARSER.add_clustering_options()
PARSER.add_training_options()

def hxnet(**kwargs):
    input_file_path = kwargs['infile']
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    logger.info(f'Readout chip: {readout}')

    if kwargs['arch'] == 'gnn':
        gnn_model = GNNRegression()
        model = ModelGNN(gnn_model)
    elif kwargs['arch'] == 'dnn':
        dnn_model = ModelDNN.load_pretrained()
        # load only the architecture without training parameters
        model_arch = model_from_json(dnn_model.model.to_json())
        model = ModelDNN(model_arch)

    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'],
                              header['pitch'], None, model)

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
    
    if kwargs['arch'] == 'gnn':
        # for gnn the best model is saved during training 
        model.train(xdata, ydata, epochs=kwargs['epochs'], name=kwargs['modelname'])
    if kwargs['arch'] == 'dnn':
        xdata = xdata[:, 0, :]
        earlystopping = EarlyStopping('val_loss', patience=10, restore_best_weights=True)
        history = model.train(xdata, ydata, epochs=kwargs['epochs'], callbacks=[earlystopping])
        model.save(kwargs['modelname'])
        ModelDNN.plot_history(history)

if __name__ == '__main__':
    hxnet(**vars(PARSER.parse_args()))
    plt.show()
