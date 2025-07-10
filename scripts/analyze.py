"""Analyze and plot reconstructed positions and energy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tables

from hexsample.analysis import fit_histogram
from hexsample.hist import Histogram1d
from hexsample.modeling import Gaussian

__description__ = \
"""Plot mc and recon position and energy distributions
"""


parser = argparse.ArgumentParser(description=__description__)
parser.add_argument('infile', help='Input file path')
parser.add_argument('--bins', default=50, help='Number of bins')
args = parser.parse_args()

def hist(args):
    infile_path = args.infile
    bins = args.bins

    with tables.open_file(infile_path, 'r') as file:
        recon_table = file.root.recon.recon_table.read()
        mc_table = file.root.mc.mc_table.read()

    # Recon energy distribution
    recon_en = recon_table['energy']
    e_binning = np.linspace(min(recon_en), max(recon_en), bins)
    recon_en_hist = Histogram1d(e_binning, xlabel='Reconstructed Energy [eV]')
    recon_en_hist.fill(recon_en)
    plt.figure('Reconstructed Energy Distribution')
    fitstatus = fit_histogram(recon_en_hist, fit_model=Gaussian, show_figure=True)

    # Recon position distr
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(recon_table['posx'], bins=bins)
    axs[0, 0].set_xticklabels([])
    axs[1, 1].hist(recon_table['posy'], bins=bins, 
                   orientation='horizontal')
    axs[1, 1].set_yticklabels([])
    axs[1, 0].hist2d(recon_table['posx'], recon_table['posy'], bins=bins)
    axs[1, 0].set_xlabel('X [cm]')
    axs[1, 0].set_ylabel('Y [cm]')
    fig.suptitle('Reconstructed Position Distribution')

    # Mc position distr
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(mc_table['absx'], bins=bins)
    axs[0, 0].set_xticklabels([])
    axs[1, 1].hist(mc_table['absy'], bins=bins, 
                   orientation='horizontal')
    axs[1, 1].set_yticklabels([])
    axs[1, 0].hist2d(mc_table['absx'], mc_table['absy'], bins=bins)
    axs[1, 0].set_xlabel('X [cm]')
    axs[1, 0].set_ylabel('Y [cm]')
    fig.suptitle('Montecarlo Position Distribution')

    # Position difference distr
    x_diff = mc_table['absx'] - recon_table['posx']
    y_diff = mc_table['absy'] - recon_table['posy']
    
    fig, ax = plt.subplots(1, 1)
    ax.hist2d(x_diff, y_diff, bins=bins)
    ax.set_xlabel('X difference [cm]')
    ax.set_ylabel('Y difference [cm]')
    fig.suptitle('Difference between mc and recon positions')

    plt.show()



if __name__ == '__main__':
    hist(args)
