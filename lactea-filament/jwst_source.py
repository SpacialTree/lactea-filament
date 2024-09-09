import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.svo_fps import SvoFps
import cutout_manager as cm
from dust_extinction.averages import CT06_MWGC

class JWSTSource():
    def __init__(self, source, name=None):
        self.ra = source['skycoord_ref'].ra
        self.dec = source['skycoord_ref'].dec
        self.catalog = source
        self.skycoord = SkyCoord(self.ra, self.dec, frame='icrs')

    def band(self, band):
        return self.catalog['mag_ab_'+band.lower()]
    
    def color(self, band1, band2):
        return self.band(band1) - self.band(band2)

    def get_Av_182410(self):
        av182410 = (self.catalog['mag_ab_f182m'] - self.catalog['mag_ab_f410m']) / (CT06_MWGC()(1.82*u.um) - CT06_MWGC()(4.10*u.um))
        #a182 = CT06_MWGC()(1.82*u.um)
        #f182mdr = basetable['mag_ab_f182m'] - a182 * av182410
        return av182410

    def get_dereddened_mag(self, band='f182m'):
        extinction = CT06_MWGC()(query_svo_fps(band)['WavelengthCen'].to(u.micron))
        return self.catalog['mag_ab_'+band.lower()] - extinction*self.get_Av()
        
    def plot_position(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.ra, self.dec, transform=ax.get_transform('world'), **kwargs)
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')
        return ax

    def plot_SED(self, filters=['f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f466n'], ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for f in filters:
            filt_data = query_svo_fps(f)
            wavelength = filt_data['WavelengthCen'].to(u.micron)
            filter_width = filt_data['WidthEff'].to(u.micron)
            flux = self.catalog['flux_jy_'+f.lower()]*u.Jy
            flux_err = self.catalog['eflux_jy_'+f.lower()]*u.Jy
            ax.errorbar(wavelength, flux*wavelength, 
                         xerr=filter_width, yerr=flux_err*wavelength,
                         color='k', fmt='o')
            ax.scatter(wavelength, flux*wavelength, marker='x', color='k')
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        ax.set_ylabel(r'Flux Density ($\mu$mJy)')
        return ax

    def get_cutout_rgb(self, cutout_size=5*u.arcsec):
        cutout_data, cutout_rgb = cm.get_cutout_rgb(self.skycoord, l=cutout_size, w=cutout_size)
        return cutout_data, cutout_rgb

    def __repr__(self):
        return f'JWSTSource({self.ra}, {self.dec})'


def query_svo_fps(filtername):
    filtername = 'JWST/NIRCam.'+filtername.upper()
    svo = SvoFps()
    ind = svo.get_filter_list(facility='JWST', instrument='NIRCam')
    return ind[ind['filterID']==filtername]