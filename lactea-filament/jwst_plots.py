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
import regions
from regions import Regions

class JWSTCatalog(Plotter):
    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

        self.coords = self.catalog['skycoord_ref']
        self.ra = self.coords.ra
        self.dec = self.coords.dec

        self.mag_ab_f405n = self.catalog['mag_ab_f405n']
        self.mag_ab_f410m = self.catalog['mag_ab_f410m']
        self.mag_ab_f466n = self.catalog['mag_ab_f466n']
        self.mag_ab_f187n = self.catalog['mag_ab_f187n']
        self.mag_ab_f182m = self.catalog['mag_ab_f182m']
        self.mag_ab_f212n = self.catalog['mag_ab_f212n']

    def color(self, band1, band2):
        return self.catalog[f'mag_ab_{band1.lower()}'] - self.catalog[f'mag_ab_{band2.lower()}']

    def band(self, band):
        return self.catalog[f'mag_ab_{band.lower()}']

    def plot_position(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.ra, self.dec, **kwargs)
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        return ax

    def get_qf_mask(self, qf=0.4):
        mas_405 = np.logical_or(np.array(self.catalog['qfit_f405n'])<qf, np.isnan(np.array(self.mag_ab_f405n)))
        mas_410 = np.logical_or(np.array(self.catalog['qfit_f410m'])<qf, np.isnan(np.array(self.mag_ab_f410m)))
        mask = np.logical_and(mas_405, mas_410)
        mas_466 = np.logical_or(np.array(self.catalog['qfit_f466n'])<qf, np.isnan(np.array(self.mag_ab_f466n)))
        mask = np.logical_and(mask, mas_466)
        mas_187 = np.logical_or(np.array(self.catalog['qfit_f187n'])<qf, np.isnan(np.array(self.mag_ab_f187n)))
        mask = np.logical_and(mask, mas_187)
        mas_182 = np.logical_or(np.array(self.catalog['qfit_f182m'])<qf, np.isnan(np.array(self.mag_ab_f182m)))
        mask = np.logical_and(mask, mas_182)
        mas_212 = np.logical_or(np.array(self.catalog['qfit_f212n'])<qf, np.isnan(np.array(self.mag_ab_f212n)))
        mask = np.logical_and(mask, mas_212)
        return mask

    def table_qf_mask(self):
        mask = self.get_qf_mask()
        return self.catalog[mask]

    def get_count_mask(self):
        mask = np.array([self.catalog[colname] < 0.1 for colname in self.catalog.colnames if colname.startswith('emag')])
        mask = mask.max(axis=0)
        return mask

    def get_region_mask(self, reg, wcs):
        mask = reg[0].contains(self.coords, wcs=wcs)
        #np.array([reg[0].contains(coord, wcs=wcs) for coord in self.coords])
        return mask

    def table_region_mask(self, reg, wcs):
        mask = self.get_region_mask(reg, wcs)
        return self.catalog[mask]