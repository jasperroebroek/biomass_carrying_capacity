from src.disturbance_regime import model_natural_disturbance_regime, model_local_disturbance_regime
from src.load_data import load_data
from src.pixel_area import define_pixel_area
from src.utils import HomogeneityFactorParams, validate_in_store, ModelParams, template_array, to_raster


def model_framework(model_params: ModelParams,
                    homogeneity_factor_params: HomogeneityFactorParams) -> None:
    """Last modelling step, finally calculating expected biomass"""
    if not model_params == homogeneity_factor_params:
        raise ValueError("Parameter sets don't match")

    model_natural_disturbance_regime(model_params,
                                     homogeneity_factor_params=homogeneity_factor_params)
    model_local_disturbance_regime(model_params,
                                   homogeneity_factor_params=homogeneity_factor_params)
    define_pixel_area(model_params.window_size)

    if validate_in_store('expected_biomass', model_params, homogeneity_factor_params):
        print("- Skipping: Natural disturbance regime")
        return

    print("- Natural disturbance regime")

    mean_prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    natural_dr = load_data('natural_disturbance_regime', model_params=model_params,
                           homogeneity_factor_params=homogeneity_factor_params)

    expected_biomass = (mean_prb * (1 - natural_dr)).values

    template = template_array(mean_prb)
    to_raster(expected_biomass, 'expected_biomass', template, model_params, homogeneity_factor_params)

    print("- Finished modelling framework!")
