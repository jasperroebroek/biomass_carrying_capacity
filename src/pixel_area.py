from pathlib import Path

import numpy as np

from src.load_data import load_data
from src.utils import template_array


def haversine(lat1, lon1, lat2, lon2):
    """ Haversine distance
    lat: y, lon: x
    """
    earth_radius = 6372.8000
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return earth_radius * c


def define_pixel_area(window_size: int) -> None:
    print("- Defining pixel area")

    biomass = load_data('biomass')
    template = template_array(biomass, window_size=window_size)
    transform = template.rio.transform(recalc=True)

    height, width = template.shape
    x_ul, y_ul = transform * transform.translation(0, 0) * (0, np.arange(height))
    x_ur, y_ur = transform * transform.translation(1, 0) * (0, np.arange(height))
    x_ll, y_ll = transform * transform.translation(0, 1) * (0, np.arange(height))
    dist_horizontal = haversine(y_ul, x_ul, y_ur, x_ur)
    dist_vertical = haversine(y_ul, x_ul, y_ll, x_ll)
    area = dist_horizontal * dist_vertical

    path = Path('processed_data', 'global', str(window_size * 250), 'pixel_area')
    path.mkdir(parents=True, exist_ok=True)

    template.data[:] = np.transpose(np.tile(area, (width, 1)))
    template.rio.to_raster(path / 'pixel_area.tif')
