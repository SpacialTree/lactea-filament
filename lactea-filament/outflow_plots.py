import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import regions
from regions import Regions
from spectral_cube import SpectralCube

default_fn = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw27.cube.I.iter1.image.pbcor.fits'
default_restfreq = 86.84696*u.GHz # ACES SiO 2-1

class OutflowPlot:
    """ 
    Class for quickly plotting outflows from astronomical data cubes.
    """
    def __init__(self, position, l=5*u.arcsec, w=5*u.arcsec, restfreq=None, cube_fn=default_fn, reg=None):
        """ 
        Parameters
        ----------
        position : astropy.coordinates.SkyCoord
            Center of the region to extract from the cube.
        l : astropy.units.Quantity, default=5*u.arcsec
            Length of the region to extract from the cube.
        w : astropy.units.Quantity, default=5*u.arcsec
            Width of the region to extract from the cube.
        restfreq : astropy.units.Quantity, optional
            Rest frequency of the spectral cube. If not provided, the rest frequency will be read from the FITS header.
        cube_fn : str, optional
            File path to the FITS file containing the spectral cube.
        """
        assert position.isscalar, "Only scalar SkyCoord objects are supported."

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
                
        if reg is not None:
            self.reg = reg
        else:
            self.reg = regions.RectangleSkyRegion(position, self.l, self.w)

    def open_cube(self):
        """ 
        Open the spectral cube file and return the cube object.
        """
        cube = SpectralCube.read(self.cube_fn).with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=self.restfreq)
        return cube

    def get_subcube(self):
        """
        Extract the subcube defined by the region.
        """
        cube = self.open_cube()
        subcube = cube.subcube_from_regions([self.reg])
        return subcube

    def get_spectral_slab(self, vmin, vmax):
        """ 
        Extract a spectral slab from the cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity
            Maximum velocity of the slab.
        """
        subcube = self.get_subcube()
        slab = subcube.spectral_slab(vmin, vmax)
        return slab

    def get_moment0(self, vmin=None, vmax=None):
        """ 
        Calculate the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        """
        if vmin is not None and vmax is not None:
            slab = self.get_spectral_slab(vmin, vmax)
            return slab.moment0()
        elif vmin is not None or vmax is not None:
            raise ValueError("Both vmin and vmax must be provided.")
        else:
            subcube = self.get_subcube()
            return subcube.moment0()

    def plot_moment0(self, vmin=None, vmax=None, ax=None, **kwargs):
        """ 
        Plot the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        """
        moment0 = self.get_moment0(vmin, vmax)
        if ax is None:
            ax = plt.subplot(projection=moment0.wcs)
        ax.imshow(moment0.value, **kwargs)

    ### Todo: Add in default Carta contour stylings
    def make_levels(self, data, nlevels=5):
        """ 
        Create linear contour levels for the moment 0 map.

        Parameters
        ----------
        data : np.ndarray
            Data to create contour levels from.
        nlevels : int, optional
            Number of contour levels to create.
        """
        data_max = np.nanmax(data)
        data_min = np.nanmin(data)
        levels = np.linspace(data_min, data_max, nlevels)
        return levels
        
    def plot_moment0_contours(self, vmin=None, vmax=None, levels=None, ax=None, **kwargs):
        """ 
        Plot contours of the moment 0 map of the spectral cube.

        Parameters
        ----------
        vmin : astropy.units.Quantity, optional
            Minimum velocity of the slab.
        vmax : astropy.units.Quantity, optional
            Maximum velocity of the slab.
        levels : list, optional
            List of contour levels to plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        """
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
        """
        Plot blue and red outflows on the moment 0 map.

        Parameters
        ----------
        vcen : astropy.units.Quantity, default=0*u.km/u.s
            Center velocity of the outflow.
        vmin : astropy.units.Quantity, default=-10*u.km/u.s
            Minimum velocity of the blueshifted outflow.
        vmax : astropy.units.Quantity, default=10*u.km/u.s
            Maximum velocity of the redshifted outflow.
        levels : list, optional
            List of contour levels to plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot the moment 0 map on.
        blue_color : str, default='blue'
            Color of the blueshifted outflow contours.
        red_color : str, default='red'
            Color of the redshifted outflow contours.
        """
        if ax is None:
            ax = plt.subplot(projection=self.get_moment0().wcs)
        # Plot redshifted outflow
        self.plot_moment0_contours(vmin=vcen, vmax=vmax, levels=levels, ax=ax, color=red_color, **kwargs)
        # Plot blueshifted outflow
        self.plot_moment0_contours(vmin=vmin, vmax=vcen, levels=levels, ax=ax, color=blue_color, **kwargs)

"""
“start-step-multiplier”

A set of “N” levels will be computed from “Start” with a (variable) “Step” and a “Multiplier”. For example, if start = 1.0, step = 0.1, N = 5, and multiplier = 2, five levels will be generated as “1.0, 1.1, 1.3, 1.7, 2.5”. The function of the multiplier is to make the step increase for each next new level. Default parameters derived from the full image statistics (per-channel) are:

    start: mean + 5 * standard deviation

    step: 4 * standard deviation

    N: 5

    multiplier: 1

"""
def start_step_multiplier(data, nlevels=5, start=None, step=None, multiplier=None):
    if start is None:
        start = np.nanmean(data) + 5*np.nanstd(data)
    if step is None:
        step = 4*np.nanstd(data)
    if multiplier is None:
        multiplier = 1
    levels = [start]
    for i in range(nlevels-1):
        levels.append(levels[-1] + step)
        step *= multiplier
    return levels

def quickplot_SiO(position, l=5*u.arcsec, w=5*u.arcsec, reg=None):
    """ 
    Quickly plot outflows from the SiO 2-1 line in the ACES data cube.
    """
    cube_fn = default_fn
    restfreq = default_restfreq
    op = OutflowPlot(position, l=l, w=w, reg=reg, cube_fn=cube_fn, restfreq=restfreq)
    op.plot_outflows()

def get_ACES_info(line, basepath='/orange/adamginsburg/jwst/cloudc/alma/ACES/'):
    spec_tab = Table.read(f'/orange/adamginsburg/jwst/cloudc/analysis/linelist.csv')
    mol = spec_tab[spec_tab['Line']==line]
    restfreq = mol['Rest (GHz)'].data[0]*u.GHz
    cube_fn = f'{basepath}/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw{spw}.cube.I.iter1.image.pbcor.fits'
    return restfreq, cube_fn

def quickplot_ACES(line, position, l=5*u.arcsec, w=5*u.arcsec, reg=None):
    """ 
    Quickly plot outflows from a line in the ACES data cube.
    """
    try: 
        restfreq, cube_fn = get_ACES_info(line)
    except:
        raise ValueError(f"Line '{line}' not found in ACES linelist.csv.")

    op = OutflowPlot(position, l=l, w=w, reg=reg, cube_fn=cube_fn, restfreq=restfreq)
    op.plot_outflows()
