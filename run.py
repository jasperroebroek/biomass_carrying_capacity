from typing import List, Tuple

from src.homogeneity_factor import parameter_optimisation_homogeneity_factor
from src.model_framework import model_framework
from src.prb_model import parameter_optimisation_max_prb
from src.utils import HomogeneityFactorParams, BiomassParams, ModelParams


def main() -> None:
    parameter_set: List[Tuple[ModelParams, HomogeneityFactorParams]] = []

    # Default experiment
    default_biomass_params = BiomassParams()
    default_model_params = ModelParams(default_biomass_params, 40, 'global', 1)
    default_homogeneity_factor_params = HomogeneityFactorParams(default_biomass_params, 40, 'global')
    parameter_set.append((default_model_params, default_homogeneity_factor_params))

    # Legacy experiment
    model_params = ModelParams(default_biomass_params, 40, 'global', 0.99)
    parameter_set.append((model_params, default_homogeneity_factor_params))

    # Final result uncertainty
    for error_scale in [-1, 1]:
        biomass_params = BiomassParams(error_scale=error_scale)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')
        parameter_set.append((model_params, homogeneity_factor_params))

    # Increase biomass by x percent
    for percent_increase in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        biomass_params = BiomassParams(scale=1 + percent_increase/100)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        parameter_set.append((model_params, default_homogeneity_factor_params))

    # Monte Carlo Simulation adding noise
    for i in range(50):
        biomass_params = BiomassParams(noise=True, seed=i)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        parameter_set.append((model_params, default_homogeneity_factor_params))

    # Run at different percentiles
    for q in [0.9, 0.95, 0.99]:
        model_params = ModelParams(default_biomass_params, 40, 'global', q)
        parameter_set.append((model_params, default_homogeneity_factor_params))

    # Biomass error multiplication
    for s in [0.5, 1, 1.5, 2, 2.5]:
        biomass_params = BiomassParams(error_scale=s)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        parameter_set.append((model_params, default_homogeneity_factor_params))

    # Run at different window sizes
    for ws in [40, 60, 80, 100]:
        model_params = ModelParams(default_biomass_params, ws, 'global', 1)
        homogeneity_factor_params = HomogeneityFactorParams(default_biomass_params, ws, 'global')
        parameter_set.append((model_params, homogeneity_factor_params))

    # Run SREX unit experiment
    model_params = ModelParams(default_biomass_params, 40, 'global', 1, srex=True)
    parameter_set.append((model_params, default_homogeneity_factor_params))

    for i, pms in enumerate(parameter_set):
        model_params, homogeneity_factor_params = pms
        print(f"{i+1} of {len(parameter_set)}", model_params, "\n", homogeneity_factor_params)
        print("#" * 50)
        model_framework(model_params, homogeneity_factor_params)


if __name__ == '__main__':
    # Hyperparameter tuning models
    parameter_optimisation_max_prb()
    parameter_optimisation_homogeneity_factor()

    # Run model and experiments
    main()
