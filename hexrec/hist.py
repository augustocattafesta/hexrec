"""Extension of hist.py from hexsample
"""

import numpy as np
import matplotlib

from hexsample.plot import plt
from hexsample.hist import HistogramBase

class Histogram2d(HistogramBase):

    """Container class for two-dimensional histograms.
    """

    PLOT_OPTIONS = dict(cmap=plt.get_cmap('hot'))

    def __init__(self, xbins: np.array, ybins: np.array, xlabel: str = '',
                 ylabel: str = '', zlabel: str = 'Entries/bin') -> None:
        """Constructor.
        """
        HistogramBase.__init__(self, (xbins, ybins), [xlabel, ylabel, zlabel])

    def _plot(self, logz: bool = False, mean: bool = False, **kwargs) -> None:
        """Overloaded make_plot() method.
        """
        x, y = (v.flatten() for v in np.meshgrid(self.bin_centers(0), self.bin_centers(1)))
        bins = self.binning
        w = self.content.T.flatten()
        if mean:
            # suppress possible divide-by-zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                w  = w / self.entries.T.flatten()
        if logz:
            # Hack for a deprecated functionality in matplotlib 3.3.0
            # Parameters norm and vmin/vmax should not be used simultaneously
            # If logz is requested, we intercent the bounds when created the norm
            # and refrain from passing vmin/vmax downstream.
            vmin = kwargs.pop('vmin', None)
            vmax = kwargs.pop('vmax', None)
            kwargs.setdefault('norm', matplotlib.colors.LogNorm(vmin, vmax))
        plt.hist2d(x, y, bins, weights=w, **kwargs)
        colorbar = plt.colorbar()
        if self.labels[2] is not None:
            colorbar.set_label(self.labels[2])
