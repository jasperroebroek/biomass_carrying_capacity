import pandas as pd
import rasterio as rio
from rasterio.mask import raster_geometry_mask

from src.load_data import load_data
from src.utils import BiomassParams, ModelParams, HomogeneityFactorParams

# Base model parameters for all plots. Some figures declare them specifically
biomass_params = BiomassParams()
model_params = ModelParams(biomass_params, 40, 'global', 1)
homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')

# Some standard files needed for calculations
forest_fraction = load_data('forest_fraction', model_params=model_params).values
pixel_area = load_data('pixel_area', resolution=model_params.window_size * 250).values


def scale(a):
    return a * forest_fraction * pixel_area * 100 * 1E-9


def aggregate_in_polygon(data, func, polygon):
    path = load_data("max_prb", model_params=model_params).encoding['source']
    mask = raster_geometry_mask(rio.open(path), polygon, invert=True)[0]
    return func(data[mask])


def load_climate_classes():
    with open("data/climate_classes/koppen_legend.txt") as f:
        data = f.read().split("\n")

    clim_legend = {0: ("Water", "Water", "Water", 0)}
    for entry in data[3:33]:
        key, string = entry.split(":")
        key = int(key)
        string = string[:string.find("[")].strip()
        clim_class = string[:string.find(" ")].strip()
        clim_descr = string[string.find(" "):].strip()
        major_class = clim_class[0]
        major_class_idx = ord(clim_class[0].lower()) - 96
        clim_legend[key] = (clim_class, clim_descr, major_class, major_class_idx)
    df_clim_legend = pd.DataFrame.from_dict(clim_legend, orient='index',
                                            columns=["clim_class", 'clim_descr', "clim_class_major",
                                                     "clim_class_major_idx"])

    return df_clim_legend
