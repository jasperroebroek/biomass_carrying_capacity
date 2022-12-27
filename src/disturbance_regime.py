import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.intact_forest import model_intact_forest
from src.load_data import load_data
from src.prb_model import model_mean_prb, define_min_samples_leaf
from src.utils import ModelParams, HomogeneityFactorParams, validate_in_store, to_raster, template_array


def model_natural_disturbance_regime(model_params: ModelParams,
                                     homogeneity_factor_params: HomogeneityFactorParams) -> None:
    if not model_params == homogeneity_factor_params:
        raise ValueError("Parameter sets don't match")

    model_mean_prb(model_params, homogeneity_factor_params)
    model_intact_forest(model_params)

    if validate_in_store('natural_disturbance_regime', model_params, homogeneity_factor_params):
        print("- Skipping: Natural disturbance regime")
        return

    print("- Natural disturbance regime")

    resolution = model_params.window_size * 250

    mean_prb = load_data("mean_prb", model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    mean_biomass = load_data("mean_biomass", model_params=model_params)
    input_data = load_data("x", resolution=resolution).sel(x=mean_prb.x, y=mean_prb.y, method='nearest')

    dv = 1 - mean_biomass.values / mean_prb.values

    natural_mask = load_data("intact_forest", model_params=model_params)
    data_mask = np.logical_and(np.isnan(input_data.values).sum(axis=0) == 0, ~np.isnan(dv))

    mask = np.logical_and(natural_mask.values, data_mask)
    X = input_data.values[:, mask].T
    y = dv[mask]

    min_samples_leaf = define_min_samples_leaf(model_params.window_size)
    RF = RandomForestRegressor(n_estimators=50, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=0)
    RF.fit(X, y)

    disturbance_regime = np.full_like(mean_prb.values, np.nan)
    disturbance_regime[data_mask] = RF.predict(input_data.values[:, data_mask].T)

    template = template_array(mean_prb)
    to_raster(disturbance_regime, 'natural_disturbance_regime', template, model_params, homogeneity_factor_params)


def model_local_disturbance_regime(model_params: ModelParams,
                                   homogeneity_factor_params: HomogeneityFactorParams) -> None:
    if not model_params == homogeneity_factor_params:
        raise ValueError("Parameter sets don't match")

    model_mean_prb(model_params, homogeneity_factor_params)

    if validate_in_store('local_disturbance_regime',
                         model_params=model_params,
                         homogeneity_factor_params=homogeneity_factor_params):
        print("- Skipping: Local disturbance regime")
        return

    print("- Local disturbance regime")

    mean_prb = load_data("mean_prb", model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    mean_biomass = load_data("mean_biomass", model_params=model_params)
    input_data = np.stack(np.meshgrid(mean_prb.x.values, mean_prb.y.values))

    dv = 1 - mean_biomass.values / mean_prb.values
    data_mask = ~np.isnan(dv)

    X = input_data[:, data_mask].T
    y = dv[data_mask]

    min_samples_leaf = define_min_samples_leaf(model_params.window_size)
    RF = RandomForestRegressor(n_estimators=50, min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=0)
    RF.fit(X, y)

    disturbance_regime = np.full_like(mean_prb.values, np.nan)
    disturbance_regime[data_mask] = RF.predict(input_data[:, data_mask].T)

    template = template_array(mean_prb)
    to_raster(disturbance_regime, 'local_disturbance_regime', template, model_params, homogeneity_factor_params)
