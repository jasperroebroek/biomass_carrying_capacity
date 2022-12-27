from typing import Any, Dict, Tuple

import numpy as np
from focal_stats.core.utils import OutputDict
from joblib import Memory, Parallel, delayed
from sklearn.metrics import mean_pinball_loss, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn_quantile import RandomForestMaximumRegressor

from src.load_data import load_data
from src.utils import parse_biomass, validate_in_store, to_raster, define_forest_mask, \
    HomogeneityFactorParams, template_array, define_rolling_slices

location = './cachedir'
memory = Memory(location, verbose=0)


def pinball_scorer(alpha: float):
    def pinball_loss(*args, **kwargs):
        return mean_pinball_loss(*args, **kwargs, alpha=alpha)

    return make_scorer(pinball_loss, greater_is_better=False)


def calc_homogeneity_factor(biomass: np.ndarray,
                            elevation: np.ndarray,
                            slope: np.ndarray,
                            aspect: np.ndarray,
                            min_samples_leaf) -> np.float64:
    mask = np.logical_and.reduce([~np.isnan(biomass),
                                  ~np.isnan(elevation),
                                  ~np.isnan(slope),
                                  ~np.isnan(aspect)])

    X = np.stack([elevation[mask], slope[mask], aspect[mask]], axis=1)
    y = biomass[mask]

    qrf = RandomForestMaximumRegressor(n_estimators=10, random_state=0, n_jobs=1, verbose=False,
                                       min_samples_leaf=min_samples_leaf)
    qrf.fit(X, y)

    return qrf.predict(X).mean() / y.max()


@memory.cache
def parameter_optimisation_homogeneity_factor(repeats: int = 100, alpha: float = 0.995) -> Dict[str, Any]:
    """
    :param repeats: number of times the optimisation is repeated
    :param alpha: pinball loss alpha. 0.99 approaches 1, but leaves room to penalise for overshooting, which
    prevents the algorithm to just pick the case in which most data possible is presented to pick the maximum from.
    If the alpha would be 1, the best possible solution would be to take the maximum of the data.
    """
    print("- Finding optimal parameters for homogeneity factor model")

    rs = np.random.RandomState(0)
    biomass = load_data('biomass')
    landcover = load_data("landcover")
    elevation = load_data('elevation')
    aspect = load_data("aspect")
    slope = load_data("slope")

    options_min_samples_leaf = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    results_min_samples_leaf = []
    result_count = 0

    while result_count < repeats:
        x = rs.randint(0, biomass.shape[0] - 40)
        y = rs.randint(0, biomass.shape[1] - 40)

        ind = np.s_[x: x + 40, y: y + 40]

        c_biomass = biomass[ind].values
        c_landcover = landcover[ind].values
        c_elevation = elevation[ind].values
        c_aspect = aspect[ind].values
        c_slope = slope[ind].values

        forest_mask = define_forest_mask(c_biomass, c_landcover)
        mask = np.logical_and.reduce([forest_mask,
                                      ~np.isnan(c_elevation),
                                      ~np.isnan(c_slope),
                                      ~np.isnan(c_aspect)])

        # Check if anything is missing for a full analysis
        if (~mask).sum() > 0:
            continue

        X = np.stack([c_elevation[mask], c_slope[mask], c_aspect[mask]], axis=1)
        y = c_biomass[mask]

        random_sample_idx = rs.choice(np.arange(len(y)), len(y), replace=False)

        gs = GridSearchCV(RandomForestMaximumRegressor(n_estimators=10, n_jobs=1, verbose=False, random_state=0),
                          param_grid={"min_samples_leaf": options_min_samples_leaf},
                          scoring=pinball_scorer(alpha=alpha),
                          n_jobs=-1)
        gs.fit(X[random_sample_idx], y[random_sample_idx])
        results_min_samples_leaf.append(gs.best_params_['min_samples_leaf'])
        print(f"{result_count + 1} of {repeats}: {gs.best_params_}")
        result_count += 1

    return {
        'min_samples_leaf': np.argmax(np.bincount(np.asarray(results_min_samples_leaf))).item()
    }


def _process_homogeneity_factor_tile(model_params: HomogeneityFactorParams,
                                     input_slice: Tuple[slice, slice],
                                     output_slice: Tuple[slice, slice],
                                     data_output: np.ndarray) -> None:
    biomass = load_data('biomass', region=model_params.region)[input_slice].values
    biomass_error = load_data("biomass_error", region=model_params.region)[input_slice].values
    landcover = load_data("landcover", region=model_params.region)[input_slice].values
    elevation = load_data('elevation', region=model_params.region)[input_slice].values
    aspect = load_data("aspect", region=model_params.region)[input_slice].values
    slope = load_data("slope", region=model_params.region)[input_slice].values

    qrf_params = parameter_optimisation_homogeneity_factor()

    biomass_parsed = parse_biomass(biomass, biomass_error, model_params.biomass_params)
    homogeneity_factor = data_output[output_slice]

    ws = model_params.window_size
    for i in range(biomass.shape[0] // ws):
        for j in range(biomass.shape[1] // ws):
            idx = np.s_[i * ws: (i + 1) * ws,
                  j * ws: (j + 1) * ws]
            out_idx = np.s_[i, j]

            c_biomass = biomass_parsed[idx]
            c_landcover = landcover[idx]
            c_forest_mask = define_forest_mask(c_biomass, c_landcover)
            c_biomass[~c_forest_mask] = np.nan

            if c_forest_mask.sum() < 0.1 * c_forest_mask.size:
                homogeneity_factor[out_idx] = np.nan
                continue

            c_elevation = elevation[idx]
            c_slope = slope[idx]
            c_aspect = aspect[idx]

            homogeneity_factor[out_idx] = calc_homogeneity_factor(c_biomass, c_elevation, c_slope, c_aspect,
                                                                  min_samples_leaf=qrf_params['min_samples_leaf'])


def model_homogeneity_factor(model_params: HomogeneityFactorParams) -> None:
    if validate_in_store('homogeneity_factor', homogeneity_factor_params=model_params):
        print("- Skipping: Homogeneity factor")
        return

    print("- Homogeneity factor")
    biomass = load_data('biomass', region=model_params.region)

    splits = {
        (72000, 162000): (20, 27),
        (18000, 18000): (15, 15)
    }

    with OutputDict(['hf'], shape=biomass.shape, reduce=True, window_size=model_params.window_size) as outputs:
        input_slices, output_slices = define_rolling_slices(biomass.shape, splits[biomass.shape],
                                                            model_params.window_size)

        Parallel(n_jobs=-1, verbose=10)(
            delayed(_process_homogeneity_factor_tile)(
                model_params=model_params, input_slice=input_slice, output_slice=output_slice, data_output=outputs['hf']
            ) for input_slice, output_slice in zip(input_slices, output_slices)
        )

    template = template_array(biomass, window_size=model_params.window_size)
    to_raster(outputs['hf'], 'homogeneity_factor', template, homogeneity_factor_params=model_params)
