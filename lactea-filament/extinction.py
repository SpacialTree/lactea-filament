import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
import regions
from regions import Regions
from astropy.coordinates import search_around_sky
from dust_extinction.averages import CT06_MWGC, I05_MWAvg, G21_MWAvg, CT06_MWLoc, RL85_MWGC, RRP89_MWGC, F11_MWGC
from scipy.spatial import KDTree
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm

from jwst_plots import make_cat_use
from jwst_plots import JWSTCatalog
import cutout_manager as cm

basepath = '/orange/adamginsburg/jwst/cloudc/'

pos = SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg))
l = 113.8*u.arcsec
w = 3.3*u.arcmin
reg = regions.RectangleSkyRegion(pos, width=l, height=w)

cutout_405 = cm.get_cutout_405(pos, w, l)
ww = cutout_405.wcs
data_405 = cutout_405.data
pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

cat_use = make_cat_use()
cat_filament = JWSTCatalog(cat_use.table_region_mask([reg], ww))
pixcoords = ww.all_world2pix(cat_filament.coords.ra, cat_filament.coords.dec, 0)
data = np.array(pixcoords).T

color_cut = 2.0

def make_grid(shape=cutout_405.data.shape):
    grid = np.empty(shape)
    grid.fill(np.nan)
    return grid

def get_pixcoords(cat, ww):
    return np.array(ww.all_world2pix(cat.coords.ra, cat.coords.dec, 0)).T

def make_kdtree(data, k=5):
    kdtr = KDTree(data)
    seps, inds = kdtr.query(data, k=[k])
    return seps, inds

def get_seps_arcsec(seps, pixel_scale):
    return seps * pixel_scale

def fill_grid(grid, data, value):
    data_rounded = np.floor(data).astype(int)
    for i in range(len(data_rounded)):
        x, y = data_rounded[i]
        grid[y, x] = value[i]
    return grid

def interpolate_grid(grid, fwhm):
    kernel = Gaussian2DKernel(fwhm)
    grid = convolve_fft(grid, kernel, nan_treatment='interpolate')
    return grid

def get_wcs(pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)), 
            l=113.8*u.arcsec, w=3.3*u.arcmin, short=False):
    reg = regions.RectangleSkyRegion(pos, width=l, height=w)
    if not short:
        cutout_405 = cm.get_cutout_405(pos, w, l)
        return cutout_405.wcs
    elif short:
        cutout_187 = cm.get_cutout_187(pos, w, l)
        return cutout_187.wcs
    else:
        return None

def make_stellar_separation_map_interp(cat=cat_filament, color_cut=2.0, ext=CT06_MWGC(),
                                       pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)), 
                                       l=113.8*u.arcsec, w=3.3*u.arcmin, fwhm=30, k=5):
    # Old! Don't use! It's very slow :)
    mask = (cat_filament.color('f182m', 'f410m') > color_cut)
    mask = np.logical_or(mask, np.isnan(cat_filament.band('f182m')) & ~np.isnan(cat_filament.band('f410m')))
    cat_use_red = JWSTCatalog(cat.catalog[mask])

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data = get_pixcoords(cat_use_red, ww)
    seps, inds = make_kdtree(data, k=k)
    seps_arcsec = get_seps_arcsec(seps, pixel_scale)

    grid = fill_grid(grid, data, seps_arcsec[:, 0].value)

    grid = interpolate_grid(grid, fwhm)

    return grid

def make_stellar_separation_map(cat=cat_filament, color_cut=2.0, ext=CT06_MWGC(),
                                pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)), 
                                l=113.8*u.arcsec, w=3.3*u.arcmin, fwhm=30, k=5):
    mask = (cat_filament.color('f182m', 'f410m') > color_cut)
    mask = np.logical_or(mask, np.isnan(cat_filament.band('f182m')) & ~np.isnan(cat_filament.band('f410m')))
    cat_use_red = JWSTCatalog(cat.catalog[mask])

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data_red = np.array(ww.all_world2pix(cat_use_red.coords.ra, cat_use_red.coords.dec, 0)).T
    kdtr_red = KDTree(data_red)

    grid_coords = np.indices(grid.T.shape)
    grid_coords = grid_coords.reshape(2, -1).T

    seps, inds = kdtr_red.query(grid_coords + 0.5, k=[k])

    grid[grid_coords[:, 1], grid_coords[:, 0]] = (seps[:, 0] * pixel_scale).value

    return grid

def make_stellar_density_map_interp(cat=cat_filament, color_cut=2.0, ext=CT06_MWGC(),
                                    pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)), 
                                    l=113.8*u.arcsec, w=3.3*u.arcmin, fwhm=30, k=5):

    grid = make_stellar_separation_map_interp(cat=cat, color_cut=color_cut, ext=ext, pos=pos, l=l, w=w, fwhm=fwhm, k=k)

    return k/grid**2

def make_stellar_density_map(cat=cat_filament, color_cut=2.0, ext=CT06_MWGC(), 
                             pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)),
                             l=113.8*u.arcsec, w=3.3*u.arcmin, fwhm=30, k=5):
    
        grid = make_stellar_separation_map(cat=cat, color_cut=color_cut, ext=ext, pos=pos, l=l, w=w, fwhm=fwhm, k=k)
    
        return k/grid**2

def make_extinction_map(cat=cat_filament, color_cut=2.0, ext=CT06_MWGC(), 
                        pos=SkyCoord('17:46:20.6290029866', '-28:37:49.5114204513', unit=(u.hour, u.deg)),
                        l=113.8*u.arcsec, w=3.3*u.arcmin, fwhm=30, k=5, Av_fill=85, reg=None):

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    mask = (cat_filament.color('f182m', 'f410m') > color_cut)
    mask = np.logical_or(mask, np.isnan(cat_filament.band('f182m')) & ~np.isnan(cat_filament.band('f410m')))
    cat_use_red = JWSTCatalog(cat.catalog[mask])
    if reg is not None:
        cat_use_red = JWSTCatalog(cat_use_red.table_region_mask(reg, ww))

    data = get_pixcoords(cat_use_red, ww)
    seps, inds = make_kdtree(data, k=k)

    Av = np.array(cat_use_red.get_Av_182410(ext=ext))
    too_red = np.isnan(np.array(cat_use_red.band('f182m'))) & ~np.isnan(np.array(cat_use_red.band('f410m')))
    Av[too_red] = Av_fill

    grid = fill_grid(grid, data, Av)

    mask_grid = cm.get_cutout_187(pos, w, l).data == 0
    grid[mask_grid] = np.nan

    grid = interpolate_grid(grid, fwhm)

    Av_const = color_cut / (ext(1.82*u.um) - ext(4.10*u.um))

    grid = grid - Av_const
    grid[grid < 0] = np.nan

    return grid

def extinction_map_error():
    pass

def get_column_density_estimate(ext_map, ww, dist=5*u.kpc, factor=2.21*10**21*u.cm**-2):
    grid_N = np.nansum(ext_map) * factor
    return grid_N


def get_mass_estimate(ext_map, ww, dist=5*u.kpc, factor=1.105*10**21*u.cm**-2, mpp=2.8*u.u):
    # N_H=2.21*10**21*u.cm**-2
    grid_N = np.nansum(ext_map) * factor
    pixel_area_physical = (ww.proj_plane_pixel_scales()[0] * dist).to(u.cm, u.dimensionless_angles())**2
    return (grid_N * pixel_area_physical * mpp).to(u.Msun)

