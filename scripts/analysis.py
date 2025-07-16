"""Analyze and plot reconstructed positions and energy
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import tables

from hexsample.hist import Histogram1d
from hexsample.analysis import fit_histogram
from hexsample.modeling import Gaussian

from hexrec.hist import Histogram2d
from hexrec.hexagon import HexagonalGrid, HexagonalLayout
from hexrec.app import ArgumentParser

__description__ = \
"""Plot mc and recon position and energy distributions
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_infile()
PARSER.add_analysis_options()

parser = argparse.ArgumentParser(description=__description__)
parser.add_argument('infile', help='Input file path')
parser.add_argument('--bins', type=int, default=50, help='Number of bins')
args = parser.parse_args()

def analyze(**kwargs):
    """Script to analyze a reconstructed file
    """
    infile_path = kwargs['infile']
    bins = kwargs['bins']

    with tables.open_file(infile_path, 'r') as file:
        recon_table = file.root.recon.recon_table.read()
        mc_table = file.root.mc.mc_table.read()

    # Create grid for hexagon center
    grid = HexagonalGrid(HexagonalLayout('ODD_R'), 304, 352, 0.005)
    x0, y0 = grid.pixel_to_world(*grid.world_to_pixel(0, 0))

    # Recon energy distr
    energy = recon_table['energy']
    xbins_en = np.linspace(min(energy), max(energy), bins)
    plt.figure('Reconstructed energy')
    h_energy = Histogram1d(xbins_en, xlabel='Energy [eV]')
    h_energy.fill(energy)
    _ = fit_histogram(h_energy, fit_model=Gaussian, show_figure=True)
    plt.tight_layout()

    xlabel_pos = 'x [cm]'
    ylabel_pos = 'y [cm]'

    # Mc position distr
    x_mc = mc_table['absx'] - x0
    y_mc = mc_table['absy'] - y0

    _, xbins_pos_mc, ybins_pos_mc = np.histogram2d(x_mc, y_mc, bins)

    plt.figure('Montecarlo position')
    h_pos_mc = Histogram2d(xbins_pos_mc, ybins_pos_mc, xlabel=xlabel_pos, ylabel=ylabel_pos)
    h_pos_mc.fill(x_mc, y_mc).plot(logz=False)
    plt.tight_layout()
    plt.figure('Montecarlo position x projection')
    h_pos_mc_x = Histogram1d(xbins=xbins_pos_mc, xlabel=xlabel_pos)
    h_pos_mc_x.fill(x_mc).plot()
    plt.tight_layout()
    plt.figure('Montecarlo position y projection')
    h_pos_mc_y = Histogram1d(xbins=ybins_pos_mc, xlabel=ylabel_pos)
    h_pos_mc_y.fill(y_mc).plot()
    plt.tight_layout()

    # Recon position distr
    x_rc = recon_table['posx'] - x0
    y_rc = recon_table['posy'] - y0

    _, xbins_pos_rc, ybins_pos_rc = np.histogram2d(x_rc, y_rc, bins)

    plt.figure('Reconstructed position')
    h_pos_rc = Histogram2d(xbins_pos_rc, ybins_pos_rc, xlabel=xlabel_pos, ylabel=ylabel_pos)
    h_pos_rc.fill(x_rc, y_rc).plot(logz=True)
    plt.tight_layout()
    plt.figure('Reconstructed position x projection')
    h_pos_rc_x = Histogram1d(xbins=xbins_pos_rc, xlabel=xlabel_pos)
    h_pos_rc_x.fill(x_rc).plot()
    plt.tight_layout()
    plt.figure('Reconstructed position y projection')
    h_pos_rc_y = Histogram1d(xbins=ybins_pos_rc, xlabel=ylabel_pos)
    h_pos_rc_y.fill(y_rc).plot()
    plt.tight_layout()

    # Distance distribution
    dist = np.sqrt((x_mc-x_rc)**2 + (y_mc-y_rc)**2)

    plt.figure('Position distance')
    h_dist = Histogram2d(xbins_pos_mc, ybins_pos_mc, xlabel_pos, ylabel_pos, 'Mean distance [cm]')
    h_dist.fill(x_mc, y_mc, weights=dist).plot(mean=True)
    plt.tight_layout()

    # Angle distribution
    phi_rc = np.arctan(y_rc/x_rc)
    mask = np.invert(np.isnan(phi_rc))

    phi_mc = np.arctan(y_mc/x_mc)[mask]
    phi_rc = phi_rc[mask]
    dphi = np.rad2deg(phi_mc - phi_rc)

    plt.figure('Angle difference')
    h_angle = Histogram2d(xbins_pos_mc, ybins_pos_mc, xlabel_pos, ylabel_pos,
                          zlabel='Mean angle difference [deg]')
    h_angle.fill(x_mc[mask], y_mc[mask], weights=dphi).plot(mean=True)
    plt.tight_layout()

if __name__ == '__main__':
    analyze(**vars(PARSER.parse_args()))
    plt.show()
