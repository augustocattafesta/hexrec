"""Analyze eta function and fit with power law
"""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from hexsample.fileio import DigiInputFileCircular
from hexsample.readout import HexagonalReadoutCircular
from hexsample.modeling import PowerLaw
from hexsample.hexagon import HexagonalLayout

from hexrec.app import ArgumentParser
from hexrec.clustering import ClusteringNN

__description__ = \
"""Analyze eta function and fit
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_infile()
PARSER.add_eta_options()

def hexeta(**kwargs):
    """Script to analyze eta function for 2 pixels events 
    from a simulation file
    """
    kwargs['nneighbors'] = 6

    input_file_path = kwargs['infile']
    input_file = DigiInputFileCircular(input_file_path)
    header = input_file.header
    args = HexagonalLayout(header['layout']), header['numcolumns'], header['numrows'],\
        header['pitch'], header['noise'], header['gain']
    readout = HexagonalReadoutCircular(*args)
    logger.info(f'Readout chip: {readout}')
    clustering = ClusteringNN(readout, kwargs['zsupthreshold'], kwargs['nneighbors'])

    pha = []
    x_pix = []
    y_pix = []
    absx = []
    absy = []
    n = []

    for i, event in tqdm(enumerate(input_file)):
        cluster = clustering.run(event)
        if cluster.size() == 2:
            pha.append(cluster.pha)
            x_pix.append(cluster.x[0])
            y_pix.append(cluster.y[0])
            n.append([cluster.x[1] - cluster.x[0], cluster.y[1] - cluster.y[0]])

            mc_event = input_file.mc_event(i)
            absx.append(mc_event.absx)
            absy.append(mc_event.absy)

    input_file.close()

    pha = np.array(pha)
    x_pix = np.array(x_pix)
    y_pix = np.array(y_pix)
    absx = np.array(absx)
    absy = np.array(absy)
    n = np.array(n)
    n = n / np.sqrt(np.sum(n**2, axis=1, keepdims=True))

    eta = pha[:, 1] / pha.sum(axis=1)
    pos = np.array([absx-x_pix, absy-y_pix]).T / header['pitch']
    dr = abs(pos[:, 0]*n[:, 0] + pos[:, 1]*n[:, 1])

    plt.figure('Scatter plot')
    plt.scatter(eta, dr, s=0.1)
    plt.xlabel(r'$\eta$')
    plt.ylabel('dr / pitch')
    plt.tight_layout()

    eta_bins = np.linspace(min(eta), max(eta), kwargs['bins'])
    y_profile = np.zeros(kwargs['bins']-1)
    y_profile_std = np.zeros(kwargs['bins']-1)
    bin_center = (eta_bins[:-1] + eta_bins[1:])/2
    edges_zip = zip(eta_bins[:-1], eta_bins[1:])

    for i, edges in enumerate(edges_zip):
        mask = (eta > edges[0]) & (eta < edges[1])
        y_true = dr[mask]

        mean = np.mean(y_true)
        mean_std = np.std(y_true)
        y_profile[i] = mean
        y_profile_std[i] = mean_std

    def fit_model(x, gamma):
        return PowerLaw().eval(x/0.5, 0.5, gamma)

    popt, pcov = curve_fit(fit_model, bin_center, y_profile, sigma=y_profile_std)

    plt.figure('profile')
    plt.errorbar(bin_center, y_profile, y_profile_std, fmt='.k', label='data')
    xx = np.linspace(0, max(eta), 1000)
    plt.plot(xx, fit_model(xx, popt[0]), '-k', label='best fit')
    plt.xlabel(r'$\eta$')
    plt.ylabel('dr/pitch')
    plt.legend()
    plt.tight_layout()

    chisq = np.sum(((y_profile - fit_model(bin_center, popt[0]))/y_profile_std)**2)
    ddof = len(y_profile)-1

    logger.info(f'gamma: {popt[0]} +- {np.sqrt(pcov)[0, 0]}')
    logger.info(f'chi / dof: {chisq:.1f} / {ddof}')

if __name__ == '__main__':
    hexeta(**vars(PARSER.parse_args()))
    plt.show()
