import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm



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


def get_cutout_405(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F405_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)
    
def get_cutout_410(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F410_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_466(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F466_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_187(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F187_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)
    
def get_cutout_182(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F182_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_212(position, l, w, basepath='/orange/adamginsburg/jwst/'):
    fn = f'{basepath}/cloudc/images/F212_reproj_merged-fortricolor.fits'
    return get_cutout(fn, position, l, w)

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
        simple_norm(rgb[:,:,0], stretch='asinh', min_cut=-1, max_cut=90)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_cut=-2, max_cut=210)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_cut=-1, max_cut=120)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_R.wcs

def get_cutout_I1(position, l, w):
    fn = '/orange/adamginsburg/cmz/glimpse_data/GLM_00000+0000_mosaic_I1.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_I2(position, l, w):
    fn = '/orange/adamginsburg/cmz/glimpse_data/GLM_00000+0000_mosaic_I2.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_I3(position, l, w):
    fn = '/orange/adamginsburg/cmz/glimpse_data/GLM_00000+0000_mosaic_I3.fits'
    return get_cutout(fn, position, l, w)

def get_cutout_I4(position, l, w):
    fn = '/orange/adamginsburg/cmz/glimpse_data/GLM_00000+0000_mosaic_I4.fits'
    return get_cutout(fn, position, l, w)

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
        simple_norm(rgb[:,:,0], stretch='asinh', min_cut=-1, max_cut=350)(rgb[:,:,0]),
        simple_norm(rgb[:,:,1], stretch='asinh', min_cut=-1, max_cut=200)(rgb[:,:,1]),
        simple_norm(rgb[:,:,2], stretch='asinh', min_cut=-1, max_cut=100)(rgb[:,:,2]),
    ]).swapaxes(0,2)

    return rgb_scaled.swapaxes(0,1), cutout_I1.wcs