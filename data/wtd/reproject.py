import geomappy as mp
from geomappy.utils import reproject_map_like
from geomappy.profile import resample_profile

M_ref = erio.Raster("data/biomass/biomass.vrt", fill_value=0)
profile = {10000: resample_profile(M_ref.profile, 1/40),
           15000: resample_profile(M_ref.profile, 1/60),
           20000: resample_profile(M_ref.profile, 1/80),
           25000: resample_profile(M_ref.profile, 1/100)}


for scale in [10000, 15000, 20000, 25000]:
    print(f"- {scale}")
    reproject_map_like(f"/Volumes/Elements SE/Projects/WUR_ecohydrological_controls/Data_raw/wtd/wtd_world.tif",
                       profile[scale],
                       f"data/wtd/{scale}/wtd.tif",
                       overwrite=True)
