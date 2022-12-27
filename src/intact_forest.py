import geopandas as gpd
from rasterio import features

from src.load_data import load_data
from src.utils import template_array, ModelParams, validate_in_store, to_raster


def model_intact_forest(model_params: ModelParams) -> None:
    if validate_in_store('intact_forest', model_params=model_params):
        print("- Skipping: Intact forest")
        return

    print("- Intact forest")

    template = template_array(load_data('biomass', region=model_params.region), model_params.window_size)
    IFL = gpd.read_file("data/intact_forest/ifl_2016.shp")

    shapes = ((geom, 1) for geom in IFL.geometry)
    natural = features.rasterize(shapes=shapes, fill=0, out_shape=template.shape,
                                 transform=template.rio.transform(recalc=True))

    to_raster(natural, 'intact_forest', template, model_params=model_params)
