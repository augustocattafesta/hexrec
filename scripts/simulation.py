"""Script to simulate events.
"""

import argparse
import numpy as np

from loguru import logger
from tqdm import tqdm

from hexsample import rng
from hexsample.readout import HexagonalReadoutMode, readout_chip
from hexsample.fileio import digioutput_class
from hexsample.hexagon import HexagonalLayout
from hexsample.mc import PhotonList
from hexsample.source import GaussianBeam, Source, SpectrumBase
from hexsample.sensor import Material, Sensor

from hexrec import HEXREC_DATA

__description__ = \
"""Simulate a list of digitized events from a monochromatic X-ray source.
"""

# creating parser
parser = argparse.ArgumentParser(description=__description__)
parser.add_argument('--outfile', default='hexrec_sim.h5',
                    help='Output file name (default: hexrec_sim.h5)')
parser.add_argument('--numevents', default=1000, type=int,
                    help='Number of events to generate (default: 1000)')
parser.add_argument('--energy', default=6000., type=float,
                    help='Energy [eV] of the monochromatic source (default: 6000)')
parser.add_argument('--noise', default=0, type=float,
                    help='Noise charge rms [electrons] (default: 0)')
args = parser.parse_args()

# define this class elsewhere
class Line(SpectrumBase):
    def __init__(self, energy):
        self.energy = energy

    def rvs(self, size):
        return np.full(size, self.energy)


def simulate(args):
    """Application main entry point.
    """
    output_file_path = HEXREC_DATA / args.outfile

    # Creating kwargs to update header, otherwise hxrecon returns errors due to problems with header
    # To be compatible with hexsample, keep all of these
    kwargs = {}
    kwargs['readoutmode'] = 'CIRCULAR'
    kwargs['layout'] = 'ODD_R'
    kwargs['numcolumns'] = 304
    kwargs['numrows'] = 352
    kwargs['pitch'] = 0.005
    kwargs['noise'] = args.noise
    kwargs['gain'] = 1

    rng.initialize(seed=None)
    spectrum = Line(args.energy)
    beam = GaussianBeam(0, 0, 0.1)
    source = Source(spectrum, beam)
    material = Material('Si', 0.116)
    sensor = Sensor(material, 0.03, 40.0)
    photon_list = PhotonList(source, sensor, args.numevents)
    readout_mode = HexagonalReadoutMode.CIRCULAR
    readout_args = 500, 0, 0
    hxsim_args = HexagonalLayout('ODD_R'), 304, 352, 0.005, args.noise, 1
    readout = readout_chip(readout_mode, *hxsim_args)
    logger.info(f'Readout chip: {readout}')
    output_file = digioutput_class(readout_mode)(output_file_path)

    output_file.update_header(**kwargs)
    logger.info('Starting the event loop...')
    for mc_event in tqdm(photon_list):
        x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
        digi_event = readout.read(mc_event.timestamp, x, y, *readout_args)
        output_file.add_row(digi_event, mc_event)
    logger.info('Done!')
    output_file.flush()
    output_file.close()

    return output_file_path

if __name__ == '__main__':
    simulate(args)
