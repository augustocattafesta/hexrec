"""Extension of hexsample hist module
"""

import numpy as np
import matplotlib

from hexsample.hist import HistogramBase
from hexsample.plot import plt, setup_gca

class CornerPlot(HistogramBase):

    """Container class for corner plot
    """
    PLOT_OPTIONS = dict(lw=1.25, alpha=0.4, histtype='stepfilled', 
                        cmap=plt.get_cmap('hot'))

    def __init__(self, xbins: np.array, ybins: np.array, xlabel: str = '', 
                 ylabel: str = '', zlabel: str = 'Entries/bin'):
        """Constructor"""
        HistogramBase.__init__(self, (xbins, ybins), [xlabel, ylabel, zlabel])

    def _plot(self, **kwargs) -> None:
        x, y = (v.flatten() for v in np.meshgrid(self.bin_centers(0), self.bin_centers(1)))
        bins = self.binning
        wx = self.content.sum(axis=0)
        wy = self.content.sum(axis=1)
        w2d = self.content.T.flatten()

        fig, axs = plt.subplots(2, 2)
        axs[0, 1].remove()
        axs[0, 0].hist(bins[0][:-1], bins=bins[0], weights=wx, **kwargs)
        axs[1, 1].hist(bins[1][:-1], bins=bins[1], weights=wy, orientation='horizontal', **kwargs)
        axs[1, 0].hist2d(x, y, bins=bins, weights=w2d, **kwargs)
