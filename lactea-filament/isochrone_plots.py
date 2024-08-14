import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
import regions
from regions import Regions
from cmd_plot import Plotter

basepath = '/home/savannahgramze/research/Research/JWST/cloudc/'
mist = Table.read(f'{basepath}/isochrones/MIST_iso_633a08f2d8bb1.iso.cmd', 
                  header_start=12, data_start=13, format='ascii', delimiter=' ', comment='#')

class Isochrone(Plotter):
    def __init__(self, table, distance=8.5*u.kpc):
        super().__init__()
        self.table = table
        self.logteff = table['log_Teff']
        self.logg = table['log_g']
        self.logL = table['log_L']
        self.M_init = table['initial_mass']
        self.M_actual = table['star_mass']
        self.age = table['log10_isochrone_age_yr']
        self.Z = table['[Fe/H]']
        self.distance = distance
        self.distance_modulus = 5*np.log10(distance.to(u.pc).value) - 5

        self.massrange = {'o': (16, np.max(self.M_init)),
                          'b': (2.1, 16),
                          'a': (1.4, 2.1),
                          'f': (1.04, 1.4),
                          'g': (0.8, 1.04),
                          #'k': (0.45, 0.8),
                          #'m': (0.08, 0.45)
                          }

        self.f405n = self.table['F405N']
        self.f410m = self.table['F410M']
        self.f466n = self.table['F466N']
        self.f187n = self.table['F187N']
        self.f182n = self.table['F182M']
        self.f212n = self.table['F212N']

    def age_selection(self, age):
        return self.table[self.age == age]

    def band(self, band):
        return self.table[band.upper()]

    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)

    
        