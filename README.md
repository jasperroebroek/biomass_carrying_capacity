
[![DOI](https://zenodo.org/badge/582606590.svg)](https://zenodo.org/badge/latestdoi/582606590)

Data processing and analysis steps
----------------------------------

Data is obtained through it's various online sources, as outlined in the supplementary text. Subsequently, data is
reprojected to the exact resolution of the Worldclim V2 data (5 arcmin). That is: the water table depth, soil moisture
and soil characteristics data are reprojected to match the Worldclim data, while the biomass and topographical data are
reprojected to exactly 40 times the resolution, in order for the windowed analysis to work. This scheme leaves most data
in approximately their original resolution or downscales. The only (marginal) upscaling occurs in the soil moisture
data: from 360 to 300 arc-seconds.

The data directory is structured in two ways. The high resolution data (7.5 arc-seconds) are split up in tiles of 40x40
degrees, following the structure of the globbiomass data. The lower resolution data are split per resolution: for
example data/soil/10000/cec.tif. Loading data from disk all occurs in src/load_data.py, so if you make modifications to
the file structure, this file needs to be adapted. The rest of the analysis should be independent of these choices.

To produce the results execute run.py. 
Subsequently, plots and summary stats can be produced with figures.py and stats.py
