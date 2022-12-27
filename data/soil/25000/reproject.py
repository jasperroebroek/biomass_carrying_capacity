from geomappy.utils import reproject_map_like
from rasterio.enums import Resampling
import os

files = [file for file in os.listdir("data/soil/raw") if "tif" in file]

for file in files:
    print(file)
    reproject_map_like(f"data/soil/raw/{file}",
                       f"data/biomass/25000/biomass.tif",
                       f"data/soil/25000/{file}",
                       verbose=False)
