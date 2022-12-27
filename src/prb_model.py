import math
import os
from typing import List, Dict, Any

import numpy as np
import seaborn as sns
import geopandas as gpd
from joblib import Memory
from matplotlib import pyplot as plt
from rasterio import features
from sklearn.model_selection import GridSearchCV
from sklearn_quantile import RandomForestMaximumRegressor

from src.biomass_reductions import model_biomass_reductions
from src.homogeneity_factor import model_homogeneity_factor, pinball_scorer
from src.load_data import load_data
from src.utils import ModelParams, validate_in_store, to_raster, HomogeneityFactorParams, template_array, BiomassParams

location = './cachedir'
memory = Memory(location, verbose=0)

FEATURE_DESCRIPTIONS = {
    'bdod': r"Bulk density [cg cm$^{-3}$]",
    'cec': r"Cation exchange capacity [mmol kg$^{-1}$]",
    'cfvo': r"Coarse fragments [cm$^{3}$ dm$^{-3}$]",
    'clay': r"Clay fraction [g kg$^{-1}$]",
    'nitrogen': r"Nitrogen [cg kg$^{-1}$]",
    'ocd': r"Organic carbon density [g dm$^{-3}$]",
    'ocs': r"Organic carbon stock [t ha$^{-1}$]",
    'phh2o': r"pH (water) [10 * -]",
    'sand': r"Sand fraction [g kg$^{-1}$]",
    'silt': r"Silt content [g kg$^{-1}$]",
    'soc': r"Soil organic carbon [dg kg$^{-1}$]",
    'bio_1': r'Annual Mean Temperature [$^\circ$C]',
    'bio_2': r'Mean Diurnal Range [$^\circ$C]',
    'bio_3': r'Isothermality [-]',
    'bio_4': r'Temperature Seasonality [$^\circ$C]',
    'bio_5': r'Max Temperature of Warmest Month [$^\circ$C]',
    'bio_6': r'Min Temperature of Coldest Month [$^\circ$C]',
    'bio_7': r'Temperature Annual Range [$^\circ$C]',
    'bio_8': r'Mean Temperature of Wettest Quarter [$^\circ$C]',
    'bio_9': r'Mean Temperature of Driest Quarter [$^\circ$C]',
    'bio_10': r'Mean Temperature of Warmest Quarter [$^\circ$C]',
    'bio_11': r'Mean Temperature of Coldest Quarter [$^\circ$C]',
    'bio_12': r'Annual Precipitation [mm]',
    'bio_13': r'Precipitation of Wettest Month [mm]',
    'bio_14': r'Precipitation of Driest Month [mm]',
    'bio_15': r'Precipitation Seasonality [-]',
    'bio_16': r'Precipitation of Wettest Quarter [mm]',
    'bio_17': r'Precipitation of Driest Quarter [mm]',
    'bio_18': r'Precipitation of Warmest Quarter [mm]',
    'bio_19': r'Precipitation of Coldest Quarter [mm]',
    'wtd': r'Water table depth [m]',
    'sm_mean': r'Annual Mean Soil Moisture [m$^{3}$ m$^{-3}$]',
    'sm_std': r'Soil Moisture Standard Deviation [m$^{3}$ m$^{-3}$]',
    'sm_cov': r'Soil Moisture Coefficient of Variation [-]',
    'sm_wettest_month': r'Soil Moisture of Wettest Month [m$^{3}$ m$^{-3}$]',
    'sm_dryest_month': r'Soil Moisture of Driest Month [m$^{3}$ m$^{-3}$]'}


def generate_max_prb_figure(qrf: RandomForestMaximumRegressor,
                            model_params: ModelParams,
                            feature_list: List[str]) -> None:
    feature_titles = (FEATURE_DESCRIPTIONS[var] for var in feature_list)
    feature_titles_short = np.asarray([feature[:feature.find("[") - 1] for feature in feature_titles])

    f, ax = plt.subplots(1, figsize=(8, 12))
    feature_importance = qrf.feature_importances_
    sort_idx = np.argsort(feature_importance)[::-1]
    feature_importance_titles = feature_titles_short[sort_idx]
    feature_importance_sorted = feature_importance[sort_idx]
    sns.barplot(x=feature_importance_sorted, y=feature_importance_titles, color="Grey", ax=ax)
    ax.set_xlabel("Score [-]")
    plt.tight_layout()

    biomass_str = str(model_params.biomass_params)
    if biomass_str == "":
        biomass_str = f"q{model_params.q * 100}"
    else:
        biomass_str += f"_q{model_params.q * 100}"

    path = os.path.join("figures",
                        model_params.region,
                        str(model_params.window_size * 250),
                        "_".join(('biomass', biomass_str)))
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, 'qrf_feature_importance.png'), dpi=300, bbox_inches='tight')
    f.clear()


@memory.cache
def parameter_optimisation_max_prb(alpha: float = 0.995) -> Dict[str, Any]:
    """
    :param alpha: pinball loss alpha. 0.99 approaches 1, but leaves room to penalise for overshooting, which
    prevents the algorithm to just pick the case in which most data possible is presented to pick the maximum from.
    If the alpha would be 1, the best possible solution would be to take the maximum of the data.
    """
    print("- Finding optimal parameters for homogeneity factor model")

    model_params = ModelParams(BiomassParams(), 40, 'global', 1)
    model_biomass_reductions(model_params)

    max_biomass = load_data("max_biomass", model_params=model_params)
    input_data = (
        load_data("x", resolution=model_params.window_size * 250)
            .sel(x=max_biomass.x, y=max_biomass.y, method='nearest')
    )

    mask = np.logical_and(np.isnan(input_data.values).sum(axis=0) == 0, ~np.isnan(max_biomass.values))

    X = input_data.values[..., mask].T
    y = max_biomass.values[mask]

    random_sample_idx = np.random.RandomState(0).choice(np.arange(len(y)), len(y), replace=False)

    options_min_samples_leaf = [10, 15, 20, 25, 50, 75, 100]
    gs = GridSearchCV(RandomForestMaximumRegressor(n_estimators=50, random_state=0, n_jobs=-1, verbose=True),
                      param_grid={"min_samples_leaf": options_min_samples_leaf},
                      scoring=pinball_scorer(alpha=alpha),
                      n_jobs=1,
                      verbose=10)
    gs.fit(X[random_sample_idx], y[random_sample_idx])
    print(gs.cv_results_)
    print(gs.best_params_)

    return {"min_samples_leaf": gs.best_params_['min_samples_leaf']}


def define_min_samples_leaf(ws: int) -> int:
    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)

    msl = parameter_optimisation_max_prb()['min_samples_leaf']
    return normal_round(msl * 40**2 / ws**2)


def _model_max_prb(model_params: ModelParams) -> None:
    print("- Max PRB")
    max_biomass = load_data("max_biomass", model_params=model_params)
    input_data = (
        load_data("x", resolution=model_params.window_size * 250)
            .sel(x=max_biomass.x, y=max_biomass.y, method='nearest')
    )

    mask = np.logical_and(np.isnan(input_data.values).sum(axis=0) == 0, ~np.isnan(max_biomass.values))

    X = input_data.values[..., mask].T
    y = max_biomass.values[mask]

    min_samples_leaf = define_min_samples_leaf(model_params.window_size)
    qrf = RandomForestMaximumRegressor(n_estimators=50,
                                       min_samples_leaf=min_samples_leaf,
                                       verbose=10,
                                       n_jobs=-1,
                                       random_state=0)
    qrf.fit(X, y)

    max_prb_predicted = np.full_like(max_biomass.values, np.nan)
    max_prb_predicted[mask] = qrf.predict(X)

    generate_max_prb_figure(qrf, model_params, input_data['variable'].values.tolist())
    to_raster(data=max_prb_predicted, var='max_prb',
              template=template_array(max_biomass),
              model_params=model_params)


def _model_max_prb_srex(model_params: ModelParams) -> None:
    print("- Max PRB SREX")

    max_biomass = load_data("max_biomass", model_params=model_params)
    input_data = (
        load_data("x", resolution=model_params.window_size * 250)
            .sel(x=max_biomass.x, y=max_biomass.y, method='nearest')
    )

    # Loading SREX polygons
    srex_polygons = gpd.read_file("data/ar5_regions/referenceRegions.shp")
    srex_polygons['value'] = srex_polygons.index

    shapes = ((geom, value) for geom, value in zip(srex_polygons.geometry, srex_polygons['value']))
    srex_raster = features.rasterize(shapes=shapes, fill=0, out_shape=max_biomass.shape,
                                     transform=max_biomass.rio.transform(recalc=True))

    mask = np.logical_and(np.isnan(input_data.values).sum(axis=0) == 0, ~np.isnan(max_biomass.values))

    X = input_data.values[..., mask].T
    y = max_biomass.values[mask]
    srex_values = srex_raster[mask]

    max_prb_predicted = np.full_like(max_biomass.values, np.nan)
    y_predicted = np.full_like(y, np.nan)

    min_samples_leaf = define_min_samples_leaf(model_params.window_size)

    for i in np.unique(srex_raster):
        print(f"SREX polygon {i}")
        srex_mask = srex_values == i
        try:
            qrf = RandomForestMaximumRegressor(n_estimators=50,
                                               min_samples_leaf=min_samples_leaf,
                                               n_jobs=-1, random_state=0)
            qrf.fit(X[srex_mask], y[srex_mask])
            y_predicted[srex_mask] = qrf.predict(X[srex_mask])
        except ValueError as e:
            print(e)
            pass

    max_prb_predicted[mask] = y_predicted

    to_raster(data=max_prb_predicted, var='max_prb',
              template=template_array(max_biomass),
              model_params=model_params)


def model_maximum_prb(model_params: ModelParams) -> None:
    model_biomass_reductions(model_params)
    if validate_in_store('max_prb', model_params):
        print("- Skipping: Max PRB")
        return

    if model_params.srex:
        _model_max_prb_srex(model_params)
    else:
        _model_max_prb(model_params)


def model_mean_prb(model_params: ModelParams,
                   homogeneity_factor_params: HomogeneityFactorParams) -> None:
    if not model_params == homogeneity_factor_params:
        raise ValueError("Parameter sets don't match")

    model_maximum_prb(model_params)
    model_homogeneity_factor(homogeneity_factor_params)

    if validate_in_store('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params):
        print("- Skipping: Mean PRB")
        return

    print("- Mean PRB")
    max_prb = load_data("max_prb", model_params=model_params)
    homogeneity_factor = load_data('hf', homogeneity_factor_params=homogeneity_factor_params)

    mean_prb = max_prb.values * homogeneity_factor.values

    to_raster(mean_prb, var='mean_prb',
              template=template_array(max_prb),
              model_params=model_params,
              homogeneity_factor_params=homogeneity_factor_params)
