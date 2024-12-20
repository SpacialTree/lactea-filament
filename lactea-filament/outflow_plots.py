import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

default_fn = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw27.cube.I.iter1.image.pbcor.fits'
default_restfreq = 86.84696*u.GHz # ACES SiO 2-1

class OutflowPlot:
    def __init__(self, position, l, w, restfreq=None, cube_fn=default_fn):
        self.position = position
        self.l = l
        self.w = w

        self.cube_fn = cube_fn

        if restfreq is not None:
            self.restfreq = restfreq
        elif cube_fn == default_fn:
            self.restfreq = default_restfreq
        else:
            header = fits.getheader(self.cube_fn)
            try:
                self.restfreq = header['RESTFRQ'] * u.Hz
            except KeyError:
                raise ValueError("RESTFRQ keyword not found in FITS header and no restfreq provided.")
                
        self.reg = regions.RectangleSkyRegion(self.position, width=self.l, height=self.w)

    def open_cube(self):
        cube = SpectralCube.read(self.cube_fn).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=self.restfreq)
        return cube

    def get_subcube(self):
        cube = self.open_cube()
        subcube = cube.subcube_from_regions([self.reg])
        return subcube

    def get_spectral_slab(self, vmin, vmax):
        subcube = self.get_subcube()
        slab = subcube.spectral_slab(vmin, vmax)
        return slab

    def get_moment0(self, vmin=None, vmax=None):
        if vmin is not None and vmax is not None:
            slab = self.get_spectral_slab(vmin, vmax)
            return slab.moment0()
        elif vmin is not None or vmax is not None:
            raise ValueError("Both vmin and vmax must be provided.")
        else:
            subcube = self.get_subcube()
            return subcube.moment0()

    def plot_moment0(self, vmin=None, vmax=None, ax=None, **kwargs):
        moment0 = self.get_moment0(vmin, vmax)
        if ax is None:
            ax = plt.subplot(projection=moment0.wcs)
        ax.imshow(moment0.value, **kwargs)

    def make_levels(self, mom0, nlevels=5):
        mom0_max = mom0.max()
        mom0_min = mom0.min()
        levels = np.linspace(mom0_min, mom0_max, nlevels)
        return levels

    def make_percentile_list(self, data, percentiles=[5, 95]):
        levels = []
        for p in percentiles:
            levels.append(np.percentile(data, p))
        return levels
        
    def plot_moment0_contours(self, vmin=None, vmax=None, levels=None, ax=None, **kwargs):
        moment0 = self.get_moment0(vmin, vmax)
        if ax is None:
            ax = plt.subplot(projection=moment0.wcs)
        if levels is None:
            levels = self.make_levels(moment0.value)
            ax.contour(moment0.value, levels=levels, transform=ax.get_transform(moment0.wcs), **kwargs)
        if levels == 'percentile':
            levels = make_percentile_list(moment0.value)
            ax.contour(moment0.value, levels=levels, transform=ax.get_transform(moment0.wcs), **kwargs)
        else:
            ax.contour(moment0.value, levels=levels, transform=ax.get_transform(moment0.wcs), **kwargs)

    def plot_outflows(self, vcen=0*u.km/u.s, vmin=-10*u.km/u.s, vmax=10*u.km/u.s, levels=None, 
                      ax=None, blue_color='blue', red_color='red', **kwargs):
        if ax is None:
            ax = plt.subplot(projection=self.get_moment0().wcs)
        # Plot redshifted outflow
        self.plot_moment0_contours(vmin=vcen, vmax=vmax, levels=levels, ax=ax, color=red_color, **kwargs)
        # Plot blueshifted outflow
        self.plot_moment0_contours(vmin=vmin, vmax=vcen, levels=levels, ax=ax, color=blue_color, **kwargs)