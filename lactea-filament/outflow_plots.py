import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

default_fn = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw27.cube.I.iter1.image.pbcor.fits'
default_restfreq = 86.84696*u.GHz # ACES SiO 2-1

class OutflowPlot:
    """ 
    Class for quickly plotting outflows from astronomical data cubes.
    """
    def __init__(self, position, l, w, restfreq=None, cube_fn=default_fn):
        """ 
        Parameters
        ----------
        position : astropy.coordinates.SkyCoord
            Center of the region to extract from the cube.
        l : astropy.units.Quantity
            Length of the region to extract from the cube.
        w : astropy.units.Quantity
            Width of the region to extract from the cube.
        restfreq : astropy.units.Quantity, optional
            Rest frequency of the spectral cube. If not provided, the rest frequency will be read from the FITS header.
        cube_fn : str, optional
            File path to the FITS file containing the spectral cube.
        """

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
        levels = np.linspace(mom0_min, mom0_max, nlevels)
        return levels

    def make_percentile_list(self, data, percentiles=[5, 95]):
        """ 
        Create percentile contour levels for the moment 0 map.

        Parameters
        ----------
        data : np.ndarray
            Data to calculate percentiles from.
        percentiles : list, optional
            List of percentiles to calculate.
        """
        levels = []
        for p in percentiles:
            levels.append(np.percentile(data, p))
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