import geomappy as mp
import os

for file in filter(lambda x: x[0] in ["S", "N"] and x.endswith("err.tif"),
                   os.listdir("/Volumes/Elements SE/Projects/ETH_PRB/Globbiomass")):
    print(file)
    coords = file.split("_")[0]
    mp.utils.reproject_map_like(os.path.join("/Volumes/Elements SE/Projects/ETH_PRB/Globbiomass", file),
                                f"data/biomass/{coords}_agb.tif", f"data/biomass_error/{coords}_agb_err.tif")
