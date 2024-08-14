import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u 
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.table import Table
from cmd_plot import Plotter

class VVVCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.ra = catalog['RAJ2000']
        self.dec = catalog['DEJ2000']

        self.catalog['Zmag'] = catalog['Z1ap1']
        self.catalog['Ymag'] = catalog['Y1ap1']
        self.catalog['Jmag'] = catalog['J1ap1']
        self.catalog['Hmag'] = catalog['H1ap1']
        self.catalog['Ksmag'] = catalog['Ks1ap1']

        self.Zmag = self.catalog['Zmag']
        self.Ymag = self.catalog['Ymag']
        self.Jmag = self.catalog['Jmag']
        self.Hmag = self.catalog['Hmag']
        self.Ksmag = self.catalog['Ksmag']

    def color(self, band1, band2):
        return self.catalog[band1] - self.catalog[band2]

    def band(self, band):
        return self.catalog[band]

    def plot_position(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.ra, self.dec, **kwargs)
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        return ax

        