import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from spectral_cube import SpectralCube
import regions
from regions import Regions

hpg = True

if hpg:
    basepath_jwst = '/orange/adamginsburg/jwst/cloudc/images/'
    basepath_glimpse = '/orange/adamginsburg/cmz/glimpse_data/'
    fn_cont_B3 = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s36_0.Sgr_A_star_sci.spw33_35.cont.I.iter1.image.tt0'
    fn_cont_B6 = ''
else:
    basepath_jwst = '/home/savannahgramze/research/Research/JWST/cloudc/images/'
    basepath_glimpse = '/home/savannahgramze/research/Research/glimpse/'

fn_405 = f'{basepath_jwst}/F405_reproj_merged-fortricolor.fits'
fn_410 = f'{basepath_jwst}/F410_reproj_merged-fortricolor.fits'
fn_466 = f'{basepath_jwst}/F466_reproj_merged-fortricolor.fits'
fn_187 = f'{basepath_jwst}/F187_reproj_merged-fortricolor.fits'
fn_182 = f'{basepath_jwst}/F182_reproj_merged-fortricolor.fits'
fn_212 = f'{basepath_jwst}/F212_reproj_merged-fortricolor.fits'
fn_I1 = f'{basepath_glimpse}/GLM_00000+0000_mosaic_I1.fits'
fn_I2 = f'{basepath_glimpse}/GLM_00000+0000_mosaic_I2.fits'
fn_I3 = f'{basepath_glimpse}/GLM_00000+0000_mosaic_I3.fits'
fn_I4 = f'{basepath_glimpse}/GLM_00000+0000_mosaic_I4.fits'

def get_cutout(filename, position, l, w, format='fits'):
    if format == 'fits':
        try: 
            hdu = fits.open(filename, ext='SCI')[0]
        except: 
            hdu = fits.open(filename)[0]
    elif format == 'casa':
        hdu = SpectralCube.read(filename, format='casa').hdu
    data = np.squeeze(hdu.data)
    head = hdu.header

    ww = WCS(head).celestial
    size = (l, w)
    cutout = Cutout2D(data, position=position, size=size, wcs=ww)
    return cutout

def get_cutout_region(position, l, w, frame='icrs'):
    if frame == 'galactic':
        return regions.RectangleSkyRegion(center=position.galactic, width=l, height=w)
    elif frame == 'icrs':
        return regions.RectangleSkyRegion(center=position.icrs, width=l, height=w)
    else:
        raise ValueError('frame must be either "icrs" or "galactic"')

def get_cutout_405(position, l, w):
    return get_cutout(fn_405, position, l, w)
    
def get_cutout_410(position, l, w):
    return get_cutout(fn_410, position, l, w)

def get_cutout_466(position, l, w):
    return get_cutout(fn_466, position, l, w)

def get_cutout_187(position, l, w):
    return get_cutout(fn_187, position, l, w)

def get_cutout_182(position, l, w):
    return get_cutout(fn_182, position, l, w)

def get_cutout_212(position, l, w):
    return get_cutout(fn_212, position, l, w)

def get_cutout_rgb(position, l, w):
    cutout_R = get_cutout_466(position, l, w)
    cutout_B = get_cutout_405(position, l, w)
    cutout_G = cutout_R.data + cutout_B.data

    rgb = np.array(
        [
            cutout_R.data,
            cutout_G,
            cutout_B.data
        ]
    ).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=210)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs

def get_cutout_rgb3(position, l, w):
    cutout_R = get_cutout_410(position, l, w)
    cutout_G = get_cutout_212(position, l, w)
    cutout_B = get_cutout_182(position, l, w)

    rgb = np.array(
        [
            cutout_R.data,
            cutout_G.data,
            cutout_B.data
        ]
    ).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=100)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs

def get_cutout_rgb_sw(position, l, w):
    cutout_R = get_cutout_212(position, l, w)
    cutout_G = get_cutout_187(position, l, w)
    cutout_B = get_cutout_182(position, l, w)

    rgb = np.array(
        [
            cutout_R.data,
            cutout_G.data,
            cutout_B.data
        ]
    ).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=100)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs

def get_cutout_rgb_lw(position, l, w):
    cutout_R = get_cutout_466(position, l, w)
    cutout_G = get_cutout_405(position, l, w)
    cutout_B = get_cutout_410(position, l, w)

    rgb = np.array(
        [
            cutout_R.data,
            cutout_G.data,
            cutout_B.data
        ]
    ).swapaxes(0,2).swapaxes(0,1)
    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-2, vmax=100)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs


def get_cutout_I1(position, l, w):
    return get_cutout(fn_I1, position, l, w)

def get_cutout_I2(position, l, w):
    return get_cutout(fn_I2, position, l, w)

def get_cutout_I3(position, l, w):
    return get_cutout(fn_I3, position, l, w)

def get_cutout_I4(position, l, w):
    return get_cutout(fn_I4, position, l, w)

def get_cutout_glimpse_rgb(position, l, w):
    cutout_I1 = get_cutout_I1(position, l, w)
    #cutout_I2 = get_cutout_I2(position, l, w)
    cutout_I3 = get_cutout_I3(position, l, w)
    cutout_I4 = get_cutout_I4(position, l, w)

    rgb = np.array(
        [
            cutout_I4.data, # R
            cutout_I3.data, # G 
            cutout_I1.data, # B
        ]
    ).swapaxes(0,2).swapaxes(0,1)

    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=350)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-1, vmax=200)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=100)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_I1.wcs

def get_cutout_glimpse_rgb_alt(position, l, w):
    cutout_I1 = get_cutout_I1(position, l, w)
    cutout_I2 = get_cutout_I2(position, l, w)
    cutout_I4 = get_cutout_I4(position, l, w)

    rgb = np.array(
        [
            cutout_I4.data, # R
            cutout_I2.data, # G 
            cutout_I1.data, # B
        ].swapaxes(0,2).swapaxes(0,1)
    )

    rgb_scaled = np.array([
        simple_norm(rgb[:,:,0], stretch='asinh', vmin=-1, vmax=350)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', vmin=-1, vmax=200)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', vmin=-1, vmax=100)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_I1.wcs

def get_alma_B3(position, l, w):
    B3_cont_fn = '/orange/adamginsburg/jwst/cloudc/alma/ACES/uid___A001_X15a0_X1a8.s36_0.Sgr_A_star_sci.spw33_35.cont.I.iter1.image.tt0'
    cutout_alma = get_cutout(B3_cont_fn, position, l, w, format='casa')
    return cutout_alma