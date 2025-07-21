"""Extension of clustering.py from hexsample
"""

# Definire la funzione di fit e un metodo per valutarla in funzione dei primi
# due valori di pha

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hexsample.digi import DigiEventSparse, DigiEventCircular, DigiEventRectangular
from hexsample.readout import HexagonalReadoutCircular
from hexsample.clustering import Cluster, ClusteringBase
from hexsample.modeling import PowerLaw

class Cluster(Cluster):
    def __init__(self, x: np.ndarray, y: np.ndarray, pha: np.ndarray, gamma: float) -> None:
        super().__init__(x, y, pha)
        self.gamma = gamma

    def fitted_position(self) -> Tuple[float, float]:
        if not self.x.shape[0] == 2:
            raise RuntimeError(f'Clusters must contain only 2 pixels')
        
        pitch = np.sqrt((self.x[0] - self.x[1])**2 + (self.y[0] - self.y[1])**2)
        n = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
        n = n / np.sqrt(n[0]**2 + n[1]**2)

        eta = self.pha[1] / self.pulse_height()
        r_fit = PowerLaw().eval(eta/0.5, 0.5, self.gamma)*pitch

        x_fit = self.x[0] + r_fit * n[0]
        y_fit = self.y[0] + r_fit * n[1]

        return x_fit, y_fit

@dataclass
class ClusteringNN(ClusteringBase):

    """Neirest neighbor clustering.

    This is a very simple clustering strategy where we use the highest pixel in
    the event as a seed, loop over the six neighbors (after the zero suppression)
    and keep the N highest pixels.

    Arguments
    ---------
    num_neighbors : int
        The number of neighbors (between 0 and 6) to include in the cluster.
    """

    num_neighbors: int
    gamma: float = None

    def run(self, event) -> Cluster:
        """Overladed method.

        .. warning::
           The loop ever the neighbors might likely be vectorized and streamlined
           for speed using proper numpy array for the offset indexes.
        """
        if isinstance(event, DigiEventSparse):
            pass
        elif isinstance(event, DigiEventCircular):
            # If the readout is circular, we want to take all the neirest neighbors.
            self.num_neighbors = HexagonalReadoutCircular.NUM_PIXELS - 1 # -1 is bc the central px is already considered
            col = [event.column]
            row = [event.row]
            adc_channel_order = [self.grid.adc_channel(event.column, event.row)]
            # Taking the NN in logical coordinates ...
            for _col, _row in self.grid.neighbors(event.column, event.row):
                col.append(_col)
                row.append(_row)
                # ... transforming the coordinates of the NN in its corresponding ADC channel ...
                adc_channel_order.append(self.grid.adc_channel(_col, _row))
            # ... reordering the pha array for the correspondance (col[i], row[i]) with pha[i].
            pha = event.pha[adc_channel_order]
            # Converting lists into numpy arrays
            col = np.array(col)
            row = np.array(row)
            pha = np.array(pha)
        # pylint: disable = invalid-name
        elif isinstance(event, DigiEventRectangular):
            seed_col, seed_row = event.highest_pixel()
            col = [seed_col]
            row = [seed_row]
            for _col, _row in self.grid.neighbors(seed_col, seed_row):
                col.append(_col)
                row.append(_row)
            col = np.array(col)
            row = np.array(row)
            pha = np.array([event(_col, _row) for _col, _row in zip(col, row)])
        # Zero suppressing the event (whatever the readout type)...
        pha = self.zero_suppress(pha)
        # Array indexes in order of decreasing pha---note that we use -pha to
        # trick argsort into sorting values in decreasing order.
        idx = np.argsort(-pha)
        # Only pick the seed and the N highest pixels.
        # This is useless for the circular readout because in that case all 
        # neighbors are used for track reconstruction.
        mask = idx[:self.num_neighbors + 1]
        # If there's any zero left in the target pixels, get rid of it.
        mask = mask[pha[mask] > 0]
        # Trim the relevant arrays.
        col = col[mask]
        row = row[mask]
        pha = pha[mask]
        x, y = self.grid.pixel_to_world(col, row)
        return Cluster(x, y, pha, self.gamma)