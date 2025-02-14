import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.coordinates import SkyCoord
import regions
from regions import Regions
from scipy.spatial import KDTree
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.nddata import Cutout2D

from spectral_cube import SpectralCube
import importlib as imp

from dust_extinction.averages import CT06_MWLoc, I05_MWAvg, CT06_MWGC, G21_MWAvg, RL85_MWGC, RRP89_MWGC, F11_MWGC

import icemodels
imp.reload(icemodels)
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.gaussian_model_components import co_ice_wls_icm, co_ice_wls, co_ice_widths, co_ice_bandstrength
from icemodels.core import optical_constants_cache_dir, read_ocdb_file, download_all_ocdb

from astroquery.svo_fps import SvoFps
from astropy import table

from jwst_plots import make_cat_use
from jwst_plots import JWSTCatalog
import extinction as ex
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

def co_ice_modeling(ref_band='f410m'):
    
    filter_data = SvoFps.get_filter_list('JWST', instrument="NIRCam")
    filter_data.add_index('filterID')
    flxd = filter_data['filterID']
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    filterid = 'JWST/NIRCam.F466N'
    xarr = np.linspace(filter_data.loc[filterid]['WavelengthMin'] * u.AA,
                    filter_data.loc[filterid]['WavelengthMax'] * u.AA,
                    1000)
    xarr = np.linspace(3.95*u.um, 4.8*u.um, 5000)
    phx4000 = atmo_model(4000, xarr=xarr)

    #aspec = absorbed_spectrum(1e18*u.cm**-2, load_molecule_ocdb('co'), spectrum=phx4000['fnu'], xarr=xarr)

    trans = SvoFps.get_transmission_data(filterid)

    # loading CO molecule
    molecule = 'co'
    # ocdb version gives nonsense
    # CO molecule constants 
        #load_molecule_ocdb(molecule) # OCDB = optical constants database
    consts = baratta_co = read_ocdb_file(f'{optical_constants_cache_dir}/1_CO_(1)_12.5K_Baratta.txt')
    # phx4000 = stellar atmosphere model spectrum at 4000K
    xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
    # column densities of CO ice
    cols = np.geomspace(1e15, 1e22, 25)
    dmags410 = []
    dmags466 = []

    #print(f"  column,   mag410,  mag410*,  mag466n, mag466n*, dmag410, dmag466")
    for col in cols:
        # for each column density of CO (ice?), make a spectrum of it 
        # absorbed_spectrum takes spectrum and puts the effects of an absorption feature in front of it 
        spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molecule_data[molecule]['molwt'],
                                spectrum=phx4000['fnu'].quantity, # flux array
                                xarr=xarr, # wavelength array
                                )
        cmd_x = (f'JWST/NIRCam.{ref_band.upper()}', 'JWST/NIRCam.F466N')
        flxd_ref = fluxes_in_filters(xarr, phx4000['fnu'].quantity)
        flxd = fluxes_in_filters(xarr, spec)
        # the star's magnitude
        mags_x_star = (-2.5*np.log10(flxd_ref[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                    -2.5*np.log10(flxd_ref[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)))
        # the magnitude of the star with the CO ice
        #mags_x = flxd[cmd_x[0]].to(u.ABmag).value, flxd[cmd_x[1]].to(u.ABmag).value
        mags_x = (-2.5*np.log10(flxd[cmd_x[0]] / u.Quantity(jfilts.loc[cmd_x[0]]['ZeroPoint'], u.Jy)),
                -2.5*np.log10(flxd[cmd_x[1]] / u.Quantity(jfilts.loc[cmd_x[1]]['ZeroPoint'], u.Jy)))
        # the difference in magnitudes in F410M and F466N
        dmags466.append(mags_x[1]-mags_x_star[1])
        dmags410.append(mags_x[0]-mags_x_star[0])
        # why would f410m change at all?
        #print(f"{col:8.1g}, {mags_x[0]:8.1f}, {mags_x_star[0]:8.1f}, {mags_x[1]:8.1f}, {mags_x_star[1]:8.1f}, {dmags410[-1]:8.1f}, {dmags466[-1]:8.1f}")

    dmag_466m410 = np.array(dmags466) - np.array(dmags410) 
    return dmag_466m410, cols

def unextinct(cat, ext, band1, band2, Av):
    return cat.color(band1, band2) + (ext(int(band1[1:-1])/100*u.um) - ext(int(band2[1:-1])/100*u.um)) * Av#cat.get_Av('f182m', 'f410m')

def make_co_column_map(cat=cat_filament, color_cut=0.5, ext=CT06_MWLoc(), pos=pos, l=l, w=w, fwhm=30, k=1, reg=None):
    # Which two bands to use for extinction correction?
    ref_band1 = 'f182m'
    ref_band2 = 'f212n'#'f410m'
    # Which band, out of F405N and F410M to use for CO column density?
    ref_band3 = 'f410m'#'f405n'#'f410m'

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = ex.make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)
    
    mask = (cat.color(ref_band1, ref_band2) > color_cut) | (np.isnan(np.array(cat.band(ref_band1))) & ~np.isnan(np.array(cat.band(ref_band2))))
    mask = mask & (cat.color(ref_band3, 'f466n') < 0)
    cat = JWSTCatalog(cat.catalog[mask])
    if reg is not None:
        cat = JWSTCatalog(cat.table_region_mask(reg, ww))
    Av = cat.get_Av(ref_band1, ref_band2, ext=ext)

    dmag_466m410, cols = co_ice_modeling(ref_band3)

    unextincted_466m410_av182410 = unextinct(cat, ext, 'f466n', ref_band3, Av=Av)
    #measured_466m410 + (CT06_MWGC()(4.66*u.um) - CT06_MWGC()(4.10*u.um)) * av182410
    cat.catalog['N(CO)'] = np.interp(unextincted_466m410_av182410, dmag_466m410[cols<1e21], cols[cols<1e21])

    data = ex.get_pixcoords(cat, ww)
    seps, inds = ex.make_kdtree(data, k=k)
    grid = ex.fill_grid(grid, data, cat.catalog['N(CO)'])

    grid_interp = ex.interpolate_grid(grid, fwhm)

    return grid_interp

def get_mass_estimate(ext_map, ww, dist=5*u.kpc, co_abundance=10**(-4), mpp=2.8*u.u):
    pixel_area_physical = (ww.proj_plane_pixel_scales()[0] * dist).to(u.cm, u.dimensionless_angles())**2
    grid_N = np.nansum(ext_map)*u.cm**(-2) * mpp * pixel_area_physical / co_abundance
    return grid_N.to(u.Msun)

def plot_Av_COice(cat=cat_filament, color_cut=2.0, ext=CT06_MWLoc(), reg=None, extras=False, label=None, **kwargs):
    #cat_red = cat.catalog[(cat.color('f182m', 'f410m') > color_cut) | (np.isnan(cat.band('f182m')) & ~np.isnan(cat.band('f410m')))]
    # Which two bands to use for extinction correction?
    ref_band1 = 'f182m'
    ref_band2 = 'f212n'#'f410m'
    # Which band, out of F405N and F410M to use for CO column density?
    ref_band3 = 'f410m'#'f405n'#'f410m'

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = ex.make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)
    
    mask = (cat.color(ref_band1, ref_band2) > color_cut) | (np.isnan(np.array(cat.band(ref_band1))) & ~np.isnan(np.array(cat.band(ref_band2))))
    mask = mask & (cat.color(ref_band3, 'f466n') < 0)
    cat_red = JWSTCatalog(cat.catalog[mask])
    if reg is not None:
        ww = ex.get_wcs()
        cat_red = JWSTCatalog(JWSTCatalog(cat_red).table_region_mask(reg, ww))
    else:
        cat_red = JWSTCatalog(cat_red)

    av212410 = cat.get_Av(ref_band2, ref_band1)
    unextincted_466m410_av212410 = unextinct(cat, ext, 'f466n', ref_band3)
    dmag_466m410, cols = co_ice_modeling()
    inferred_co_column_av212410 = np.interp(unextincted_466m410_av212410, dmag_466m410[cols<1e21], cols[cols<1e21])

    #fig = plt.figure(figsize=(8,6))
    plt.scatter(av212410, inferred_co_column_av212410, label=f'Av {ref_band1}/{ref_band2}', **kwargs)

    NCOofAV = 2.21e21 * np.linspace(0.1, 100, 1000) * 1e-4
    plt.xlim(-5, 95)
    plt.ylim(2e15, 5e20)
    #pl.plot([10, 35], [1e17, 1e20], 'k--', label='log N = 0.12 A$_V$ + 15.8');

    if extras:
        ## by eye fit Filament
        #x1,y1 = 25,2e17
        #x2,y2 = 40,5e18
        #m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
        #b = np.log10(y1 / 10**(m * x1))
        #plt.plot([x1, x2], 10**np.array([x1*m+b, x2*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Filament]')

        # by eye fit Filament
        x1,y1 = 22,1e17
        x2,y2 = 50,7e18
        m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
        b = np.log10(y1 / 10**(m * x1))
        plt.plot([x1, x2], 10**np.array([x1*m+b, x2*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Filament]')

        # by-eye fit Brick
        x1,y1 = 10,2e17
        x2,y2 = 43,1e19
        x1,y1 = 33,8e16
        x2,y2 = 80,3e19
        m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
        b = np.log10(y1 / 10**(m * x1))
        plt.plot([x1, x2], 10**np.array([x1*m+b, x2*m+b]), 'b--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f} [Ginsburg 2023 Brick]')
        plt.plot([7, 23], [0.5e17, 7e17], 'g', label='log N = 0.07 A$_V$ + 16.2 [BGW 2015]')
        plt.plot(np.linspace(0.1, 100, 1000), NCOofAV,
            label='100% of CO in ice if N(H$_2$)=2.2$\\times10^{21}$ A$_V$', color='r', linestyle=':')
    
    plt.xlabel("A$_V$ from [F182M]-[F410M]")
    plt.ylabel("N(CO) ice\nfrom Hudgins 1993 constants,\n4000K Phoenix atmosphere")
    plt.yscale('log')
    plt.legend()

    return av212410, inferred_co_column_av212410

def plot_ice_CCD(cat, ax=None, bins=100, threshold=5, cmap='autumn_r', color='k', s=1):
    import mpl_plot_templates as template
    x = np.array(cat.color('f182m', 'f212n'))
    x[x>5] = np.nan
    y = np.array(cat.color('f410m', 'f466n'))
    y[y>5] = np.nan
    if ax is None:
        ax = plt.subplot()
    
    template.adaptive_param_plot(x, y, threshold=threshold, bins=bins, cmap=cmap, marker_color=color, markersize=s, axis=ax)
    plt.xlabel('F182M - F212N')
    plt.ylabel('F410M - F466N');