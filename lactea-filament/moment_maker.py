from spectral_cube import SpectralCube
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import numpy as np
from astropy.io import fits
from astropy import units as u

def get_moment0(restfreq, vmin, vmax, filename):
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