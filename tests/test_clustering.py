"""Test of clustering.py
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from hexsample.hexagon import HexagonalLayout

from hexrec.clustering import Cluster
from hexrec.hexagon import HexagonalGrid
from hexrec.network import ModelGNN, ModelDNN

def test_cluster_centroid(size=10):
    """Test centroid position reconstruction
    """
    pitch = 0.005
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 10, 10, pitch)

    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(0, 0))
    x = [x0]
    y = [y0]

    for _col, _row in grid.neighbors(*grid.world_to_pixel(x0, y0)):
        _x, _y = grid.pixel_to_world(_col, _row)
        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)

    peak = 1000
    pha = np.sort(np.random.randint(0, peak, (size, len(x))))[:, ::-1]

    x_centroid = np.zeros(size)
    y_centroid = np.zeros(size)
    for i, _pha in enumerate(pha):
        cluster = Cluster(x, y, _pha)
        x_centroid[i], y_centroid[i] = cluster.centroid()

    plt.figure('Centroid reconstruction')
    plt.scatter(x_centroid, y_centroid, s=1)
    plt.xlabel('X position [cm]')
    plt.xlabel('Y position [cm]')

def test_cluster_eta(size=10):
    """Test eta event reconstruction and compare with centroid reconstruction.
    """
    pitch = 0.005
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 10, 10, pitch)

    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(0, 0))
    x = [x0]
    y = [y0]

    for _col, _row in grid.neighbors(*grid.world_to_pixel(x0, y0))[:1]:
        _x, _y = grid.pixel_to_world(_col, _row)
        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)

    peak = 1000
    pha = np.sort(np.random.randint(0, peak, (size, len(x))))[:, ::-1]
    eta = pha[:, 1] / (pha[:, 0] + pha[:, 1])

    eta_position = np.zeros((size, 2))
    centroid_position = np.zeros((size, 2))
    for i, _pha in enumerate(pha):
        cluster = Cluster(x, y, _pha, pitch=pitch, gamma=0.25)
        eta_position[i] = cluster.eta_position()
        centroid_position[i] = cluster.centroid()

    dr_fit = np.sqrt((eta_position[:, 0] - x[0])**2 + (eta_position[:, 1] - y[0])**2)
    dr_cent = np.sqrt((centroid_position[:, 0] - x[0])**2 + (centroid_position[:, 1] - y[0])**2)
    plt.figure('Eta and centroid comparison')
    plt.scatter(eta, dr_fit/pitch, s=1, color='k', label='fit')
    plt.scatter(eta, dr_cent/pitch, s=1, color='b', label='centroid')
    plt.xlabel('eta')
    plt.ylabel('dr / pitch')
    plt.legend()

def test_cluster_nnet(size=10):
    """Test neural network position reconstruction
    """
    pitch = 0.005
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 10, 10, pitch)

    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(0, 0))
    x = [x0]
    y = [y0]

    for _col, _row in grid.neighbors(*grid.world_to_pixel(x0, y0)):
        _x, _y = grid.pixel_to_world(_col, _row)
        x.append(_x)
        y.append(_y)

    x = np.array(x)
    y = np.array(y)

    peak = 1000
    pha = np.sort(np.random.randint(0, peak, (size, len(x))))[:, ::-1]

    gnn = ModelGNN.load_pretrained()
    dnn = ModelDNN.load_pretrained()

    gnn_position = []
    dnn_position = []
    for i, _pha in enumerate(pha):
        cluster_gnn = Cluster(x, y, _pha, pitch=pitch, model=gnn)
        cluster_dnn = Cluster(x, y, _pha, pitch=pitch, model=dnn)

        gnn_position.append(cluster_gnn.nnet_position())
        dnn_position.append(cluster_dnn.nnet_position())

    gnn_position = np.array(gnn_position)
    dnn_position = np.array(dnn_position)
    
    plt.figure('Neural network reconstruction')
    plt.scatter(gnn_position[:, 0], gnn_position[:, 1], color='k', s=1, label='gnn')
    plt.scatter(dnn_position[:, 0], dnn_position[:, 1], color='b', s=1, label='dnn')
    plt.xlabel('X position [cm]')
    plt.ylabel('Y position [cm]')
    plt.legend()

if __name__ == '__main__':
    test_cluster_centroid(size=10000)
    test_cluster_eta(size=1000)
    test_cluster_nnet(size=1000)
    plt.show()
