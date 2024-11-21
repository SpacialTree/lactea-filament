from spectral_cube import SpectralCube
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.table import Table

basepath = '/orange/adamginsburg/jwst/cloudc/'

def get_percentile_list(data, percents=[98, 99, 99.9, 99.99]):
    #[87, 95, 99, 99.5, 99.9, 99.99]
    arr = []
    for per in percents:
        arr.append(np.nanpercentile(data, per))
    return np.array(arr)

def get_mom0(restfreq, vmin, vmax, filename):
    cube = SpectralCube.read(filename, format='fits')
    subcube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=restfreq).spectral_slab(vmin, vmax)
    subcube.allow_huge_operations = True
    mom0 = subcube.to(u.K).moment0()
    mom0.allow_huge_operations = True
    return mom0

def get_mom0_cutout(pos, l, w, mom0):
    data = np.squeeze(mom0.data)
    head = mom0.header
    ww = WCS(head).celestial
    size = (l, w)

    cutout = Cutout2D(data, position=pos, size=size, wcs=ww)
    return cutout

def get_ACES_mom0(pos, l, w, line, vmin, vmax):
    spec_tab = Table.read(f'{basepath}/analysis/linelist.csv')
    mol = spec_tab[spec_tab['Line']==line]
    restfreq = mol['Rest (GHz)'].data[0]*u.GHz
    spw = mol['12m SPW'].data[0]
    cube_fn = f'{basepath}/alma/ACES/uid___A001_X15a0_X1a8.s38_0.Sgr_A_star_sci.spw{spw}.cube.I.iter1.image.pbcor.fits'

    mom0 = get_mom0(restfreq, vmin, vmax, cube_fn)
    mom0_cutout = get_mom0_cutout(pos, l, w, mom0)

    return mom0_cutout

def plot_ACES_mom0(pos, l, w, line, vmin, vmax):
    mom0_cutout = get_mom0_box(pos, l, w, line, vmin, vmax)

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection=mom0_cutout.wcs)
    im = ax.imshow(mom0_cutout.data, cmap='Greys', vmin=0)

    ax.set_xlabel('Right Ascension', fontsize=14)
    ax.set_ylabel('Declination', fontsize=14)
    ax.set_title(line, fontsize=16)

    cbar = plt.colorbar(im)
    cbar.set_label('Integrated Intensity (K km/s)', fontsize=14)
    plt.tight_layout()


    return mom0_cutout