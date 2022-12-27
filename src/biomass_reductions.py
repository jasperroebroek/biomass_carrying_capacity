import warnings
from typing import Dict, Tuple

import focal_stats as fs
import numpy as np
from focal_stats.core.utils import OutputDict
from joblib import Parallel, delayed

from src.load_data import load_data
from src.utils import parse_biomass, to_raster, validate_in_store, define_forest_mask, ModelParams, \
    template_array, define_rolling_slices


def _process_biomass_reductions(model_params: ModelParams,
                                input_slice: Tuple[slice, slice],
                                output_slice: Tuple[slice, slice],
                                data_output: Dict[str, np.ndarray]) -> None:
    warnings.filterwarnings('ignore')
    biomass = load_data('biomass', region=model_params.region)[input_slice].values
    biomass_error = load_data("biomass_error", region=model_params.region)[input_slice].values
    landcover = load_data("landcover", region=model_params.region)[input_slice].values

    biomass_parsed = parse_biomass(biomass, biomass_error, model_params.biomass_params)
    forest_mask = define_forest_mask(biomass_parsed, landcover)
    biomass_parsed[~forest_mask] = np.nan

    data_output['mean_forest_biomass'][output_slice] = fs.focal_mean(biomass_parsed,
                                                                     window_size=model_params.window_size,
                                                                     reduce=True, fraction_accepted=0.1)
    data_output['forest_fraction'][output_slice] = fs.rolling.rolling_mean(forest_mask,
                                                                           window_size=model_params.window_size,
                                                                           reduce=True)
    data_output['std_forest_biomass'][output_slice] = fs.focal_std(biomass_parsed,
                                                                   window_size=model_params.window_size,
                                                                   reduce=True, fraction_accepted=0.1)

    if model_params.q == 1:
        max_forest_biomass = fs.focal_max(biomass_parsed,
                                          window_size=model_params.window_size,
                                          reduce=True, fraction_accepted=0.1)
    else:
        max_forest_biomass = np.nanquantile(fs.rolling.rolling_window(biomass_parsed,
                                                                      window_size=model_params.window_size,
                                                                      reduce=True), q=model_params.q, axis=(2, 3))
        max_forest_biomass[np.isnan(data_output['mean_forest_biomass'][output_slice])] = np.nan

    data_output['max_forest_biomass'][output_slice] = max_forest_biomass


def model_biomass_reductions(model_params: ModelParams) -> None:
    if (
            validate_in_store('mean_forest_biomass', model_params) and
            validate_in_store('max_forest_biomass', model_params) and
            validate_in_store('forest_fraction', model_params) and
            validate_in_store('std_forest_biomass', model_params)
    ):
        print("- Skipping: Biomass reductions")
        return

    print("- Biomass reductions")
    biomass = load_data('biomass', region=model_params.region)

    splits = {
        (72000, 162000): (20, 27),
        (18000, 18000): (15, 15)
    }

    with OutputDict(['mean_forest_biomass', 'max_forest_biomass', 'forest_fraction', 'std_forest_biomass'],
                    shape=biomass.shape, reduce=True, window_size=model_params.window_size) as outputs:
        input_slices, output_slices = define_rolling_slices(biomass.shape, splits[biomass.shape],
                                                            model_params.window_size)

        Parallel(n_jobs=-1, verbose=10)(
            delayed(_process_biomass_reductions)(
                model_params=model_params, input_slice=input_slice, output_slice=output_slice, data_output=outputs
            ) for input_slice, output_slice in zip(input_slices, output_slices)
        )

    template = template_array(biomass, window_size=model_params.window_size)
    to_raster(outputs['mean_forest_biomass'], 'mean_forest_biomass', template, model_params)
    to_raster(outputs['max_forest_biomass'], 'max_forest_biomass', template, model_params)
    to_raster(outputs['forest_fraction'], 'forest_fraction', template, model_params)
    to_raster(outputs['std_forest_biomass'], 'std_forest_biomass', template, model_params)
