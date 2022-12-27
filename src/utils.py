import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import focal_stats as fs
import numpy as np
import xarray as xr


@dataclass
class BiomassParams:
    intercept: float = 0
    scale: float = 1
    error_scale: float = 0
    noise: bool = False
    seed: int = 0

    def __str__(self):
        parts = []
        if self.intercept != 0:
            parts.append(self.intercept)

        if self.scale != 1:
            parts.append(f"{self.scale}b")

        if self.noise:
            parts.append(f'noise_{self.seed}')
        elif self.error_scale != 0:
            parts.append(f"{self.error_scale}e")

        return "-".join(parts).replace(".", "_")


@dataclass
class ParamsBase:
    biomass_params: BiomassParams
    window_size: int
    region: str

    def __eq__(self, other):
        if (
                self.window_size == other.window_size and
                self.region == other.region
        ):
            return True
        return False


@dataclass
class ModelParams(ParamsBase):
    q: float
    srex: bool = False


class HomogeneityFactorParams(ParamsBase):
    pass


def parse_biomass(biomass: np.ndarray,
                  biomass_error: np.ndarray,
                  biomass_params: BiomassParams) -> np.ndarray:
    if biomass_params.noise:
        e = np.random.RandomState(seed=biomass_params.seed).normal(scale=biomass_error)
    else:
        e = biomass_params.error_scale * biomass_error

    b = biomass_params.intercept + biomass_params.scale * biomass + e
    return b


def define_forest_mask(biomass: np.ndarray, landcover: np.ndarray) -> np.ndarray:
    return np.logical_and.reduce([biomass > 0, landcover > 100, landcover < 200])


def template_array(t: xr.DataArray, window_size: int = 1) -> xr.DataArray:
    if window_size == 1:
        x = t.x.values
        y = t.y.values
    else:
        x = fs.rolling_mean(t.x.values, window_size=window_size, reduce=True)
        y = fs.rolling_mean(t.y.values, window_size=window_size, reduce=True)

    arr = (
        xr.DataArray(
            data=np.nan,
            coords={'y': y, 'x': x},
            dims=['y', 'x'])
        ).rio.write_crs("EPSG:4326", inplace=True)

    return arr


def define_path(var: str,
                model_params: Optional[ModelParams] = None,
                homogeneity_factor_params: Optional[HomogeneityFactorParams] = None) -> Path:
    var_list_left_branch = ['mean_forest_biomass', 'max_forest_biomass', 'forest_fraction', 'max_prb', 'intact_forest',
                            'std_forest_biomass']
    var_list_right_branch = ['homogeneity_factor']
    var_list_combined_branches = ['mean_prb', 'natural_disturbance_regime', 'local_disturbance_regime',
                                  'expected_biomass']

    if var not in var_list_right_branch + var_list_combined_branches + var_list_left_branch:
        raise ValueError(f"var not recognised: {var}")

    if model_params is not None and homogeneity_factor_params is not None:
        if model_params != homogeneity_factor_params:
            raise ValueError(f"Parameter sets do not match:"
                             f"\nModel params: {model_params}"
                             f"\nHF params: {homogeneity_factor_params}")

    if model_params is None and homogeneity_factor_params is not None:
        parts = ['processed_data',
                 homogeneity_factor_params.region,
                 str(homogeneity_factor_params.window_size * 250),
                 var]
    elif model_params is not None:
        resolution = str(model_params.window_size * 250)
        if model_params.srex:
            resolution += "_srex"

        parts = ['processed_data',
                 model_params.region,
                 resolution,
                 var]
    else:
        raise TypeError("Neither model_params or homogeneity_factor params is present")

    if var in var_list_left_branch + var_list_combined_branches:
        if model_params is None:
            raise TypeError(f"model_params not provided but necessary for {var}")

        biomass_str = str(model_params.biomass_params)
        if biomass_str == "":
            biomass_str = f"q{model_params.q * 100}"
        else:
            biomass_str += f"_q{model_params.q * 100}"

        parts.append("_".join(('biomass', biomass_str)))

    if var in var_list_right_branch + var_list_combined_branches:
        if homogeneity_factor_params is None:
            raise TypeError(f"model_params not provided but necessary for {var}")

        biomass_str = str(homogeneity_factor_params.biomass_params)
        if biomass_str == "":
            parts.append('homogeneity_factor')
        else:
            parts.append("_".join(('homogeneity_factor', biomass_str)))

    parts.append(f"{var}.tif")
    return Path(os.path.join(*parts))


def to_raster(data: np.ndarray,
              var: str,
              template: xr.DataArray,
              model_params: Optional[ModelParams] = None,
              homogeneity_factor_params: Optional[HomogeneityFactorParams] = None) -> None:
    path = define_path(var, model_params, homogeneity_factor_params)
    path.parent.mkdir(parents=True, exist_ok=True)

    da = template.copy(deep=True)
    da.data[:] = data
    da.rio.to_raster(str(path))


def validate_in_store(var: str,
                      model_params: Optional[ModelParams] = None,
                      homogeneity_factor_params: Optional[HomogeneityFactorParams] = None) -> bool:
    path = define_path(var, model_params, homogeneity_factor_params)
    return path.exists()


def generate_slices(shape: Tuple[int, int],
                    split: Tuple[int, int]) -> Tuple[Tuple[slice, slice]]:
    step_y = shape[0] // split[0]
    step_x = shape[1] // split[1]

    if step_y != shape[0] / split[0]:
        raise ValueError("Split Y is not exact")

    if step_x != shape[1] / split[1]:
        raise ValueError("Split X is not exact")

    slices = tuple(
        (slice(sy - step_y, sy), slice(sx - step_x, sx))
        for sy in range(step_y, shape[0] + 1, step_y)
        for sx in range(step_x, shape[1] + 1, step_x)
    )

    return slices


def define_rolling_slices(shape: Tuple[int, int],
                          split: Tuple[int, int],
                          window_size: int) -> Tuple[Tuple[Tuple[slice, slice]], Tuple[Tuple[slice, slice]]]:
    if shape[0] // split[0] != shape[0] / split[0]:
        raise ValueError("Window_size Y is not exact")

    if shape[1] // split[1] != shape[1] / split[1]:
        raise ValueError("Window_size X is not exact")

    input_slices = generate_slices(shape, split)
    output_slices = generate_slices((shape[0] // window_size, shape[1] // window_size), split)

    return input_slices, output_slices
