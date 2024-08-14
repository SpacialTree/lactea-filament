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
from dust_extinction.averages import RRP89_MWGC, CT06_MWGC, F11_MWGC
from dust_extinction.parameter_averages import CCM89

basepath = '/home/savannahgramze/research/Research/JWST/cloudc/'
mist = Table.read(f'{basepath}/isochrones/MIST_iso_633a08f2d8bb1.iso.cmd', 
                  header_start=12, data_start=13, format='ascii', delimiter=' ', comment='#')

class Isochrone(Plotter):
    def __init__(self, table, age, distance=8.5*u.kpc, mass_min=0.05, Av=30.0):
        super().__init__()
        self.age = age
        self.age_sel = table['log10_isochrone_age_yr'] == self.age
        self.mass_min = mass_min
        self.mass_sel = table['star_mass'] >= mass_min
        self.sel = np.logical_and(self.age_sel, self.mass_sel)

        self.table = table[self.sel]
        
        self.logteff = table['log_Teff'][self.sel]
        self.logg = table['log_g'][self.sel]
        self.logL = table['log_L'][self.sel]
        self.M_init = table['initial_mass'][self.sel]
        self.M_actual = table['star_mass'][self.sel]
        self.ages = table['log10_isochrone_age_yr'][self.sel]
        self.Z = table['[Fe/H]'][self.sel]

        self.distance = distance
        self.distance_modulus = 5*np.log10(distance.to(u.pc).value) - 5
        self.ext = CT06_MWGC()
        self.Av = Av

        self.table['F405N'] = self.table['F405N'] + self.distance_modulus + self.Av*self.ext(4.05*u.micron)
        self.table['F410M'] = self.table['F410M'] + self.distance_modulus + self.Av*self.ext(4.10*u.micron)
        self.table['F466N'] = self.table['F466N'] + self.distance_modulus + self.Av*self.ext(4.66*u.micron)
        self.table['F187N'] = self.table['F187N'] + self.distance_modulus + self.Av*self.ext(1.87*u.micron)
        self.table['F182M'] = self.table['F182M'] + self.distance_modulus + self.Av*self.ext(1.82*u.micron)
        self.table['F212N'] = self.table['F212N'] + self.distance_modulus + self.Av*self.ext(2.12*u.micron)

    def band(self, band):
        return self.table[band.upper()]

    def color(self, band1, band2):
        return self.band(band1.upper()) - self.band(band2.upper())

    def plot_isochrone(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color('f187n', 'f405n'), self.table['F187N'], **kwargs)
        ax.set_xlabel('F187N - F405N')
        ax.set_ylabel('F187N')
        return ax

        