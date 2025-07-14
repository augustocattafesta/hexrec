"""Test of analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tables

from hexrec import HEXREC_DATA
from hexrec.hist import Histogram2d
from hexrec.hexagon import HexagonalGrid, HexagonalLayout


def test_histogram_heatmap():
    infile_path = HEXREC_DATA / 'hexagonal_sim_recon.h5'

    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 304, 352, 0.005)
    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(0, 0))

    with tables.open_file(infile_path, 'r') as file:
        recon_table = file.root.recon.recon_table.read()
        mc_table = file.root.mc.mc_table.read()

    x_rc = recon_table['posx'] - x0
    y_rc = recon_table['posy'] - y0

    x_mc = mc_table['absx'] - x0
    y_mc = mc_table['absy'] - y0

    r_mc = np.sqrt(x_mc**2 + y_mc**2)
    r_rc = np.sqrt(x_rc**2 + y_rc**2)

    dr = r_mc - r_rc

    _, xbins, ybins = np.histogram2d(x_mc, y_mc, bins=(100, 100))
    # sums, _, _ = np.histogram2d(x, y, weights=z, bins=bins)
    h = Histogram2d(xbins, ybins, 'X [cm]', 'Y [cm]', 'dr [cm]')
    h.fill(x_mc, y_mc, weights=dr)
    plt.figure()
    plt.title('Histogram taking mean value of each pixel')
    h.plot(mean=True)
    plt.figure()
    plt.title('Histogram not taking mean value of each pixel')
    h.plot(mean=False)

if __name__ == '__main__':
    test_histogram_heatmap()
    plt.show()