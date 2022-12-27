import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Protocol

import numpy as np
import rioxarray as rxr
import xarray as xr

from src.utils import ModelParams, HomogeneityFactorParams, define_path


def load_flat_file(f: Union[str, Path], cache: bool = True, name: str = 'data') -> Union[xr.DataArray, xr.Dataset]:
    da = (
        rxr.open_rasterio(f, masked=True, parse_coordinates=True, cache=cache)
            .sel(band=1)
            .drop_vars("band")
            .rename(name)
    )

    return da


def load_file(f: Union[str, Path], cache: bool = True, name: str = 'data') -> Union[xr.DataArray, xr.Dataset]:
    da = (
        rxr.open_rasterio(f, masked=True, parse_coordinates=True, cache=cache)
            .rename(name)
    )

    return da


class Loader(Protocol):
    def load(self) -> xr.DataArray:
        ...


@dataclass
class HighResloader:
    region: str = 'global'


class BiomassLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/biomass/biomass.vrt", cache=False)
        else:
            return load_flat_file(f"data/biomass/{self.region}_agb.tif")


class BiomassErrorLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/biomass_error/biomass_error.vrt", cache=False)
        else:
            return load_flat_file(f"data/biomass_error/{self.region}_agb_err.tif")


class AspectLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/aspect/aspect.vrt", cache=False)
        else:
            return load_flat_file(f"data/aspect/{self.region}_aspect.tif")


class SlopeLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/slope/slope.vrt", cache=False)
        else:
            return load_flat_file(f"data/slope/{self.region}_slope.tif")


class ElevationLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/elevation/elevation.vrt", cache=False)
        else:
            return load_flat_file(f"data/elevation/{self.region}_elevation.tif")


class LandcoverLoader(HighResloader):
    def load(self) -> xr.DataArray:
        if self.region == 'global':
            return load_flat_file("data/landcover/landcover.vrt", cache=False)
        else:
            return load_flat_file(f"data/landcover/{self.region}_landcover.tif")


@dataclass
class LowResLoader:
    resolution: int = 10000


class SoilLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        soil_params = ('bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'ocd', 'ocs', 'phh2o', 'sand', 'silt', 'soc')
        rasters = [
            (load_file(f"data/soil/{self.resolution}/{param}.tif", name=param)
                    * np.asarray([5, 10, 15, 30, 40, 100])[:, None, None])
                .sum(dim='band') / 200
            for param in soil_params
        ]
        return xr.merge(rasters).to_array()


class SoilMoistureLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        soil_moisture_params = ('sm_mean', 'sm_std', 'sm_cov', 'sm_wettest_month', 'sm_dryest_month')
        rasters = [
            load_flat_file(f"data/soil_moisture/{self.resolution}/{param}.tif", name=param)
            for param in soil_moisture_params
        ]
        return xr.merge(rasters).to_array()


class WTDLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        return load_flat_file(f'data/wtd/{self.resolution}/wtd.tif')


class ClimateLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        indices = np.arange(1, 20)
        rasters = [
            load_flat_file(f"data/climate/{self.resolution}/bio_{ind}.tif", name=f"bio_{ind}")
            for ind in indices
        ]
        return xr.merge(rasters).to_array()


class XLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        soil = load_data('soil', resolution=self.resolution)
        climate = load_data('climate', resolution=self.resolution)
        wtd = load_data("wtd", resolution=self.resolution).expand_dims({'variable': ['wtd']})
        soil_moisture = load_data('soil_moisture', resolution=self.resolution)
        return xr.concat([soil, climate, wtd, soil_moisture], dim='variable')


class PixelAreaLoader(LowResLoader):
    def load(self) -> xr.DataArray:
        return load_flat_file(os.path.join('processed_data',
                                           'global',
                                           str(self.resolution),
                                           'pixel_area',
                                           'pixel_area.tif'))


@dataclass
class ReductionLoader:
    model_params: ModelParams


class IntactForestLoader(ReductionLoader):
    def load(self) -> xr.DataArray:
        path = define_path('intact_forest', self.model_params)
        return load_flat_file(path, name='intact_forest')


class MaxBiomassLoader(ReductionLoader):
    def load(self) -> xr.DataArray:
        path = define_path('max_forest_biomass', self.model_params)
        return load_flat_file(path, name='max_forest_biomass')


class MeanBiomassLoader(ReductionLoader):
    def load(self) -> xr.DataArray:
        path = define_path('mean_forest_biomass', self.model_params)
        return load_flat_file(path, name='mean_forest_biomass')


class ForestFractionLoader(ReductionLoader):
    def load(self) -> xr.DataArray:
        path = define_path('forest_fraction', self.model_params)
        return load_flat_file(path, name='forest_fraction')


@dataclass
class HomogeneityFactorLoader:
    homogeneity_factor_params: HomogeneityFactorParams

    def load(self) -> xr.DataArray:
        path = define_path('homogeneity_factor', homogeneity_factor_params=self.homogeneity_factor_params)
        return load_flat_file(path, name='forest_fraction')


@dataclass
class MaxPRBLoader:
    model_params: ModelParams

    def load(self) -> xr.DataArray:
        path = define_path('max_prb', self.model_params)
        return load_flat_file(path, name='max_prb')


@dataclass
class MatchedBranchLoader:
    model_params: ModelParams
    homogeneity_factor_params: HomogeneityFactorParams


class MeanPRBLoader(MatchedBranchLoader):
    def load(self) -> xr.DataArray:
        path = define_path('mean_prb', self.model_params, self.homogeneity_factor_params)
        return load_flat_file(path, name='mean_prb')


class NaturalDRLoader(MatchedBranchLoader):
    def load(self) -> xr.DataArray:
        path = define_path('natural_disturbance_regime', self.model_params, self.homogeneity_factor_params)
        return load_flat_file(path, name='natural_disturbance_regime')


class LocalDRLoader(MatchedBranchLoader):
    def load(self) -> xr.DataArray:
        path = define_path('local_disturbance_regime', self.model_params, self.homogeneity_factor_params)
        return load_flat_file(path, name='local_disturbance_regime')


class ExpectedBiomassLoader(MatchedBranchLoader):
    def load(self) -> xr.DataArray:
        path = define_path('expected_biomass', self.model_params, self.homogeneity_factor_params)
        return load_flat_file(path, name='expected_biomass')


def get_data_loader(var: str, **kwargs) -> Loader:
    LOADERS = {
        'biomass': BiomassLoader,
        'biomass_error': BiomassErrorLoader,
        'aspect': AspectLoader,
        'slope': SlopeLoader,
        'elevation': ElevationLoader,
        'landcover': LandcoverLoader,
        'soil': SoilLoader,
        'soil_moisture': SoilMoistureLoader,
        'wtd': WTDLoader,
        'climate': ClimateLoader,
        'max_biomass': MaxBiomassLoader,
        'mean_biomass': MeanBiomassLoader,
        'forest_fraction': ForestFractionLoader,
        'hf': HomogeneityFactorLoader,
        'x': XLoader,
        'max_prb': MaxPRBLoader,
        'mean_prb': MeanPRBLoader,
        'intact_forest': IntactForestLoader,
        'natural_disturbance_regime': NaturalDRLoader,
        'local_disturbance_regime': LocalDRLoader,
        'expected_biomass': ExpectedBiomassLoader,
        'pixel_area': PixelAreaLoader
    }

    return LOADERS[var](**kwargs)


def load_data(var: str, **kwargs):
    loader = get_data_loader(var, **kwargs)
    return loader.load()
