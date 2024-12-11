import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import astropy.units as u 
from astropy.coordinates import SkyCoord
import regions
from regions import Regions

from spectral_cube import SpectralCube
import importlib as imp

from dust_extinction.averages import CT06_MWLoc, I05_MWAvg, CT06_MWGC, G21_MWAvg, RL85_MWGC, RRP89_MWGC, F11_MWGC

import icemodels
imp.reload(icemodels)
from icemodels import absorbed_spectrum, absorbed_spectrum_Gaussians, convsum, fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data
from icemodels.gaussian_model_components import co_ice_wls_icm, co_ice_wls, co_ice_widths, co_ice_bandstrength
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

def get_dmags():

    # Filter set up 
    filter_data = SvoFps.get_filter_list('JWST', instrument="NIRCam")
    filter_data.add_index('filterID')
    flxd = filter_data['filterID']
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    # Stellar Atmosphere Model
    filterid = 'JWST/NIRCam.F466N'
    xarr = np.linspace(filter_data.loc[filterid]['WavelengthMin'] * u.AA,
                    filter_data.loc[filterid]['WavelengthMax'] * u.AA,
                    1000)
    xarr = np.linspace(3.95*u.um, 4.8*u.um, 5000)
    phx4000 = atmo_model(4000, xarr=xarr)

    # CO Ice Absorption Spectrum
    aspec = absorbed_spectrum(1e18*u.cm**-2, load_molecule('co'), spectrum=phx4000['fnu'], xarr=xarr)
    plt.plot(xarr.to(u.um), aspec / aspec.max())

    trans = SvoFps.get_transmission_data(filterid)

    # loading CO molecule
    molecule = 'co'
    # ocdb version gives nonsense
    # CO molecule constants 
    consts = load_molecule(molecule) # OCDB = optical constants database
    # phx4000 = stellar atmosphere model spectrum at 4000K
    xarr = phx4000['nu'].quantity.to(u.um, u.spectral())
    # column densities of CO ice
    cols = np.geomspace(1e15, 1e22, 25)
    dmags410 = []
    dmags466 = []

    print(f"  column,   mag410,  mag410*,  mag466n, mag466n*, dmag410, dmag466")
    for col in cols:
        # for each column density of CO (ice?), make a spectrum of it 
        # absorbed_spectrum takes spectrum and puts the effects of an absorption feature in front of it 
        spec = absorbed_spectrum(col*u.cm**-2, consts, molecular_weight=molecule_data[molecule]['molwt'],
                                spectrum=phx4000['fnu'].quantity, # flux array
                                xarr=xarr, # wavelength array
                                )
        cmd_x = ('JWST/NIRCam.F410M', 'JWST/NIRCam.F466N')
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
        print(f"{col:8.1g}, {mags_x[0]:8.1f}, {mags_x_star[0]:8.1f}, {mags_x[1]:8.1f}, {mags_x_star[1]:8.1f}, {dmags410[-1]:8.1f}, {dmags466[-1]:8.1f}")

    return cols, dmags410, dmags466

def unextinct_466m410(cat, ext=CT06_MWLoc()):
    av182410 = cat.get_Av('f182m', 'f410m')
    measured_466m410 = cat.color('f466n', 'f410m')
    unextincted_466m410_av212410 = measured_466m410 + (ext()(4.66*u.um) - ext()(4.10*u.um)) * av212410
    return unextincted_466m410_av212410

def get_co_colum(cat, ext=CT06_MWLoc()):
    cols, dmags410, dmags466 = get_dmags()
    dmag_466m410 = np.array(dmags466) - np.array(dmags410) 

    unextincted_466m410_av212410 = unextinct_466m410(cat, ext=ext)

    inferred_co_column_av212410 = np.interp(unextincted_466m410_av212410, dmag_466m410[cols<1e21], cols[cols<1e21])

    return inferred_co_column_av212410

def make_co_column_map(cat=cat_filament, color_cut=2.0, ext=CT06_MWLoc(), pos=pos, l=l, w=w, fwhm=30, k=5):
    mask = (cat.color('f182m', 'f410m') > color_cut) | (np.isnan(cat.band('f182m')) & ~np.isnan(cat.band('f410m')))
    mask = mask & (cat.color('f466n', 'f410m') < 0)
    cat_use_red = JWSTCatalog(cat.catalog[mask])

    cutout_405 = cm.get_cutout_405(pos, w, l)
    grid = ex.make_grid(cutout_405.data.shape)
    ww = cutout_405.wcs
    pixel_scale = ww.proj_plane_pixel_scales()[0] * u.deg.to(u.arcsec)

    data = ex.get_pixcoords(cat_use_red, ww)
    seps, inds = ex.get_separations(data, grid, k=k)

    co_column = get_co_colum(cat_use_red, ext=ext)

    grid = ex.fill_grid(grid, data, co_column)
    grid = ex.interpolate_grid(grid, fwhm)

    return grid