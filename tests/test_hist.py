"""Test for hist.py"""

import numpy as np
import matplotlib.pyplot as plt

from hexrec.hist import CornerPlot
from hexsample.hist import Histogram2d

def test_cornerplot(size = 100000, bins=100):
    x = np.random.normal(0, 1, size=size)
    y = np.random.normal(0, 1, size=size)
    xbins = np.linspace(min(x), max(x), bins)
    ybins = np.linspace(min(y), max(y), bins)

    
    cp = CornerPlot(xbins, ybins, 'X', 'Y')
    # cp.fill(x, y).plot()
    h2d = Histogram2d(xbins, ybins, 'x', 'y')
    h2d.fill(x,y)
    h2d.plot()




if __name__ == '__main__':
    test_cornerplot()
    plt.show()