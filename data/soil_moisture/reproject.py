from geomappy.utils import reproject_map_like
import geomappy as mp
from rasterio.enums import Resampling
import os
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np


for t in ["sm_mean", "sm_std", "sm_cov", "sm_dryest_month", "sm_wettest_month"]:
    print(t)
    for i in range(1, 5):
        print(f"- {i}")
        f = mp.Raster(f"data/soil_moisture/raw/{t}_{i}.nc")
        profile = f.profile
        profile['crs'] = "EPSG:4326"
        profile['transform'] = rio.warp.calculate_default_transform("EPSG:4326", "EPSG:4326", f.width, f.height,
                                                                    -180, -90, 180, 90)[0]
        dst = mp.Raster(f"data/soil_moisture/raw/reproj_{t}_{i}.tif", mode='w', profile=profile, overwrite=True)
        dst[:, 1800:] = f[:, :1800]
        dst[:, :1800] = f[:, 1800:]
        dst.close()
        mp.Raster.close(verbose=False)

        for res in [5000, 10000, 15000, 20000, 25000]:
            reproject_map_like(f"data/soil_moisture/raw/reproj_{t}_{i}.tif",
                               f"data/biomass/{res}/biomass.tif",
                               f"data/soil_moisture/{res}/{t}_{i}.tif",
                               verbose=False)

    for res in [5000, 10000, 15000, 20000, 25000]:
        a = np.asarray([mp.Raster(f"data/soil_moisture/{res}/{t}_{i}.tif").values for i in range(1, 5)])
        dst = mp.Raster(f"data/soil_moisture/{res}/{t}.tif", mode='w',
                        profile=mp.Raster(f"data/soil_moisture/{res}/sm_mean_1.tif").profile,
                        overwrite=True)
        dst[None] = (np.dot(a.T, [7, 21, 72, 189]).T / 289).astype(np.float32)
        mp.Raster.close()

for t in ["sm_mean", "sm_std", "sm_cov", "sm_dryest_month", "sm_wettest_month"]:
    for i in range(1, 5):
        try:
            os.remove(f"data/soil_moisture/raw/reproj_{t}_{i}.tif")
        except:
            pass
        for res in [5000, 10000, 15000, 20000, 25000]:
            try:
                os.remove(f"data/soil_moisture/{res}/{t}_{i}.tif")
            except:
                pass
