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
    def __init__(self, table, distance=8.5*u.kpc, age=10):
        super().__init__()
        self.age_sel = table['log10_isochrone_age_yr'] == age
        self.table = table[self.age_sel]
        
        self.logteff = table['log_Teff'][self.age_sel]
        self.logg = table['log_g'][self.age_sel]
        self.logL = table['log_L'][self.age_sel]
        self.M_init = table['initial_mass'][self.age_sel]
        self.M_actual = table['star_mass'][self.age_sel]
        self.ages = table['log10_isochrone_age_yr'][self.age_sel]
        self.Z = table['[Fe/H]'][self.age_sel]
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

        self.f405n = self.table['F405N'][self.age_sel]
        self.f410m = self.table['F410M'][self.age_sel]
        self.f466n = self.table['F466N'][self.age_sel]
        self.f187n = self.table['F187N'][self.age_sel]
        self.f182m = self.table['F182M'][self.age_sel]
        self.f212n = self.table['F212N'][self.age_sel]

    def band(self, band):
        return self.table[band.upper()]

    def color(self, band1, band2):
        return self.band(band1.upper()) - self.band(band2.upper())

    def plot_isochrone(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color('f187n', 'f405n'), self.f187n+self.distance_modulus, **kwargs)
        ax.set_xlabel('F187N - F405N')
        ax.set_ylabel('F187N')
        return ax
        