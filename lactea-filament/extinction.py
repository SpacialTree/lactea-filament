import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits

from jwst_plots import make_cat_use

def star_density_color(tbl, ww, dx=1, blur=False, plot=True):
    pos_smudge = SkyCoord('17:46:23.6171497910', '-28:36:43.0690114397', unit=(u.hourangle, u.deg), frame='icrs')
    size = (2.55*u.arcmin, 8.4*u.arcmin) # approx size of field
    bins_ra = np.arange(0, size[1].to(u.arcsec).value, dx)
    bins_dec = np.arange(0, size[0].to(u.arcsec).value, dx)

    bins_pix_ra = bins_ra/ww.proj_plane_pixel_scales()[1].to(u.arcsec).value
    bins_pix_dec= bins_dec/ww.proj_plane_pixel_scales()[1].to(u.arcsec).value

    crds_pix = np.array(ww.world_to_pixel(tbl['skycoord_ref']))

    if plot:
        plt.figure(figsize=(18, 6))
        ax = plt.subplot(111, projection=ww)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
    h, xedges, yedges = np.histogram2d(crds_pix[0], crds_pix[1], bins=[bins_pix_ra, bins_pix_dec])
    if not blur:
        if plot:
            h1 = ax.imshow(h.swapaxes(0,1))
            plt.colorbar(h1)
        #h1, xedges1, yedges1, y = ax.hist2d(crds_pix[0], crds_pix[1], bins=[bins_pix_ra, bins_pix_dec])
        #plt.colorbar(h1)
        return h
    elif blur:
        blurred = gaussian_filter(h, 1)
        if plot:
            h1 = ax.imshow(blurred.swapaxes(0,1))
            plt.colorbar(h1)
        #im = ax.imshow(blurred.swapaxes(0,1))
        #plt.colorbar(im)
        return blurred

def make_wcs(h_noshort, hdu):
    wcs_dict = {
        'SIMPLE' : 'T',
        'BITPIX' : -64,
        'NAXIS' : 2,
        'NAXIS1' : h_noshort.shape[0],
        'NAXIS2' : h_noshort.shape[1],
        'WCSAXES' : 2,
        'CRPIX1' : h_noshort.shape[0]/2,
        'CRPIX2' : h_noshort.shape[1]/2,
        'CDELT1' : -(2*u.arcsec).to(u.deg).value,
        'CDELT2' : (2*u.arcsec).to(u.deg).value,
        'CROTA2' : 354.6-270,
        'CUNIT1' : 'deg',
        'CUNIT2' : 'deg',
        'CTYPE1' : 'RA---TAN',
        'CTYPE2' : 'DEC--TAN',
        'CRVAL1' : hdu['SCI'].header['CRVAL1'],
        'CRVAL2' : hdu['SCI'].header['CRVAL2'],
        #'LONPOL' : 180.0,
        #'LATPOL' : 0.0,
        #'MJDREF' : 0.0,
        'BUNIT' : '# Stars/px'
    }
    input_wcs = WCS(wcs_dict)
    return input_wcs

def construct_cube(tbl, ww, hdu, dx=1, blur=False, plot=True):
    tbl_noshort = tbl[(tbl['qf_f405n']>0.6) & ~(np.isnan(tbl['mag_ab_f410m'])) & (np.isnan(tbl['mag_ab_f182m'])) & 
                      (np.isnan(tbl['mag_ab_f187n'])) & (np.isnan(tbl['mag_ab_f212n']))]
    h_noshort = star_density_color(tbl_noshort, ww, dx=dx, blur=blur)
    
    tbl_use = tbl[(tbl['qf_f405n']>0.6) & ~(np.isnan(tbl['mag_ab_f410m'])) & ~(np.isnan(tbl['mag_ab_f182m']))]
    cube = np.array([star_density_color(tbl_use[(color > lowmag) & (color < highmag)], ww, dx=dx, blur=blur) for lowmag, highmag in color_couples])

    cube_full = np.concatenate([cube, h_noshort.reshape((1,h_noshort.shape[0],h_noshort.shape[1]))])
    input_wcs = make_wcs(h_noshort, hdu)
    hdu_cube = fits.PrimaryHDU(data=cube_full.swapaxes(1,2), header=input_wcs.to_header())

    return hdu_cube

def make_cube():
    # Open file for WCS information
    fn_405 = f'{basepath}/images/F405_reproj_merged-fortricolor.fits'
    hdu = fits.open(fn_405)
    ww = WCS(fits.open(fn_405)[0].header)

    # Open catalog file
    cat_use = make_cat_use()

    hdu_cube = make_cube(tbl, ww, hdu, dx=1, blur=False, plot=True)
    hdu_cube.writeto(f'{basepath}/images/pseudo_extinction_cube.fits', overwrite=True)