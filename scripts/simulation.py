"""Script to simulate events.
"""

import argparse

from loguru import logger
from tqdm import tqdm

from hexsample import rng
from hexsample.readout import HexagonalReadoutMode, readout_chip
from hexsample.fileio import digioutput_class
from hexsample.mc import PhotonList
from hexsample.source import GaussianBeam, Source
from hexsample.sensor import Material, Sensor

from hexrec import HEXREC_DATA
from hexrec.source import Line, TriangularBeam, HexagonalBeam
from hexrec.hexagon import HexagonalLayout, HexagonalGrid
from hexrec.app import ArgumentParser

__description__ = \
"""Simulate a list of digitized events from a monochromatic X-ray source.
"""

PARSER = ArgumentParser(description=__description__)
PARSER.add_numevents(1000)
PARSER.add_outfile(HEXREC_DATA / 'sim.h5')
PARSER.add_line_source_options()
PARSER.add_simple_readout_options()
# PARSER.add_sensor_options()



def simulate(**kwargs):
    """Application main entry point.
    """
    output_file_path = HEXREC_DATA / kwargs['outfile']

    rng.initialize(seed=None)
    grid_args = HexagonalLayout(kwargs['layout']), kwargs['numcolumns'], kwargs['numrows'], kwargs['pitch']
    
    if kwargs['beamshape'].lower() == 'gaussian':
        beam = GaussianBeam(kwargs['srcposx'], kwargs['srcposy'], kwargs['srcsigma'])
    if kwargs['beamshape'] == 'triangular':
        grid = HexagonalGrid(*grid_args)
        target_col, target_row = grid.world_to_pixel(kwargs['srcposx'], kwargs['srcposy'])
        center, v0, v1 = grid.find_vertices(target_col, target_row, kwargs['trngindex'])
        beam = TriangularBeam(*center, tuple(v0), tuple(v1))
    if kwargs['beamshape'] == 'hexagonal':
        grid = HexagonalGrid(*grid_args)
        target_col, target_row = grid.world_to_pixel(kwargs['srcposx'], kwargs['srcposy'])
        center, v0, v1 = grid.find_vertices(target_col, target_row)
        beam = HexagonalBeam(*center, tuple(v0), tuple(v1))

    spectrum = Line(kwargs['energy'])
    source = Source(spectrum, beam)
    material = Material('Si', 0.116)
    sensor = Sensor(material, 0.03, 40.0)
    photon_list = PhotonList(source, sensor, kwargs['numevents'])
    readout_mode = HexagonalReadoutMode(kwargs['readoutmode'])
    readout_args = 500, 0, 0

    hxsim_args = *grid_args, kwargs['noise'], 1
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
    simulate(**vars(PARSER.parse_args()))
