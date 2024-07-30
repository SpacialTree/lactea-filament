from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import numpy as np
import matplotlib.pyplot as plt

def get_catalog(catalog_name, coord, w=1.0*u.arcmin, l=1.0*u.arcmin):  #, max_star=50):
    """ Get Catalog
    
    Function to get catalog at a given coordinate and radius using Vizier.

    Args:
        catalog_name (str): Name of the catalog.
        coord (SkyCoord): Coordinates of the center of the search.
        radius (Quantity): Radius of the search.

    Returns:
        Table: Table of the catalog.

    """
    #try:
    Vizier.ROW_LIMIT = 5e4
    #Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/348/vvv2'])[0]
    guide = Vizier.query_region(coordinates=coord, width=w, height=l, catalog=[catalog_name])[0]
    #except:
    #    print("No catalog found.")
    #    return False

    #if len(guide) > max_star:
    #    for c in guide.colnames:
    #        if 'mag' in c:
    #            guide = guide[~np.isnan(np.array(guide[c]))]
    #    if len(guide) <= max_star:
    #        return guide
    #    else:
    #        guide = guide[0:max_star]

    return guide

def get_VVV_catalog_circ(coord, radius=1.0*u.arcmin):  #, max_star=50):
    """ Get VVV Catalog
    
    Function to get VVV catalog at a given coordinate and radius using Vizier.

    Args:
        coord (SkyCoord): Coordinates of the center of the search.
        radius (Quantity): Radius of the search.

    Returns:
        Table: Table of the catalog.

    """
    # Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/348/vvv2'])[0]
    return get_catalog('II/376', coord, w=radius, l=radius)

def get_VVV_catalog(coord, w=1.0*u.arcmin, l=1.0*u.arcmin):  #, max_star=50):
    """ Get VVV Catalog
    
    Function to get VVV catalog at a given coordinate and radius using Vizier.

    Args:
        coord (SkyCoord): Coordinates of the center of the search.
        radius (Quantity): Radius of the search.

    Returns:
        Table: Table of the catalog.

    """
    # Vizier.query_region(coordinates=coord, width=width, height=height, catalog=['II/348/vvv2'])[0]
    return get_catalog('II/376', coord, w=w, l=l)

