"""Test of hist.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tables

from hexrec import HEXREC_DATA
from hexrec.hist import Histogram2d
from hexrec.hexagon import HexagonalGrid, HexagonalLayout


def test_histogram_heatmap():
    """Test for Histogram2d class
    """
    size = 1000000
    bins = 50
    x = np.random.uniform(-1, 1, size)
    y = np.random.uniform(-1, 1, size)
    z = np.exp(-x**2)*np.exp(-y**2)

    _, xbins, ybins = np.histogram2d(x, y, bins=bins)
    
    plt.figure('Events distribution')
    h = Histogram2d(xbins, ybins, 'X', 'Y')
    h.fill(x, y).plot()

    plt.figure('Function heatmap (no mean)')
    h = Histogram2d(xbins, ybins, 'X', 'Y', 'sum of f(x, y)')
    h.fill(x, y, weights=z).plot(mean=False)

    plt.figure('Function heatmap (mean)')
    h = Histogram2d(xbins, ybins, 'X', 'Y', 'f(x, y)')
    h.fill(x, y, weights=z).plot(mean=True)

if __name__ == '__main__':
   test_histogram_heatmap()
   plt.show()
