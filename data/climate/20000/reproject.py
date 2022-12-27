from geomappy.utils import reproject_map_like
from rasterio.enums import Resampling
import os

files = [file for file in os.listdir("data/climate/raw/10000") if "bio" in file]
names = [file[file.find("bio"):] for file in files]

for file, new_file in zip(files, names):
    print(file)
    reproject_map_like(f"data/climate/raw/10000/{file}",
                       f"data/biomass/20000/biomass.tif",
                       f"data/climate/20000/{new_file}")
