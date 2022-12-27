import warnings

import geopandas as gpd
import numpy as np
from scipy.stats import mode
from shapely.geometry import Point
from sklearn.metrics import r2_score

from plotting_helper_functions import load_climate_classes, aggregate_in_polygon, scale
from src.load_data import load_data, load_flat_file
from src.utils import BiomassParams, ModelParams, HomogeneityFactorParams

warnings.filterwarnings("ignore")

# Base model parameters for all plots. Some figures declare them specifically
biomass_params = BiomassParams()
model_params = ModelParams(biomass_params, 40, 'global', 1)
homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')

biomass_params_low = BiomassParams(error_scale=-1)
model_params_low = ModelParams(biomass_params_low, 40, 'global', 1)
homogeneity_factor_params_low = HomogeneityFactorParams(biomass_params_low, 40, 'global')

biomass_params_high = BiomassParams(error_scale=1)
model_params_high = ModelParams(biomass_params_high, 40, 'global', 1)
homogeneity_factor_params_high = HomogeneityFactorParams(biomass_params_high, 40, 'global')

# Loading data
IFL = gpd.read_file("data/intact_forest/ifl_2016.shp")
IFL['value'] = 1

t = forest_biomass = load_data('mean_biomass', model_params=model_params)
forest_biomass = load_data('mean_biomass', model_params=model_params).values
forest_biomass_low = load_data('mean_biomass', model_params=model_params_low).values
forest_biomass_high = load_data('mean_biomass', model_params=model_params_high).values

mean_prb = load_data("mean_prb", model_params=model_params,
                     homogeneity_factor_params=homogeneity_factor_params).values
mean_prb_low = load_data("mean_prb", model_params=model_params_low,
                         homogeneity_factor_params=homogeneity_factor_params_low).values
mean_prb_high = load_data("mean_prb", model_params=model_params_high,
                          homogeneity_factor_params=homogeneity_factor_params_high).values

forest_fraction = load_data("forest_fraction", model_params=model_params).values
pixel_area = load_data("pixel_area", resolution=model_params.window_size * 250).values
intact_forest = load_data("intact_forest", model_params=model_params).values == 1
managed_areas = ~intact_forest

natural_disturbance_regime = load_data("natural_disturbance_regime", model_params=model_params,
                                       homogeneity_factor_params=homogeneity_factor_params).values
natural_disturbance_regime_low = load_data("natural_disturbance_regime", model_params=model_params_low,
                                           homogeneity_factor_params=homogeneity_factor_params_low).values
natural_disturbance_regime_high = load_data("natural_disturbance_regime", model_params=model_params_high,
                                            homogeneity_factor_params=homogeneity_factor_params_high).values

local_disturbance_regime = load_data("local_disturbance_regime", model_params=model_params,
                                     homogeneity_factor_params=homogeneity_factor_params).values
local_disturbance_regime_low = load_data("local_disturbance_regime", model_params=model_params_low,
                                         homogeneity_factor_params=homogeneity_factor_params_low).values
local_disturbance_regime_high = load_data("local_disturbance_regime", model_params=model_params_high,
                                          homogeneity_factor_params=homogeneity_factor_params_high).values

expected_biomass = load_data("expected_biomass", model_params=model_params,
                             homogeneity_factor_params=homogeneity_factor_params).values
expected_biomass_low = load_data("expected_biomass", model_params=model_params_low,
                                 homogeneity_factor_params=homogeneity_factor_params_low).values
expected_biomass_high = load_data("expected_biomass", model_params=model_params_high,
                                  homogeneity_factor_params=homogeneity_factor_params_high).values

climate = load_flat_file("data/climate_classes/climate_classes.tif").values

df_clim_legend = load_climate_classes()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.loc[world.name == "Norway", "iso_a3"] = "NOR"
world.loc[world.name == "France", "iso_a3"] = "FRA"
world.loc[world.name == "N. Cyprus", "iso_a3"] = "CYP"
world.loc[world.name == "Kosovo", "iso_a3"] = "KOS"
world.loc[world.name == "Somaliland", "iso_a3"] = "SOM"

# INTEGRATING DATA
expected_biomass_total_PG = scale(expected_biomass)
expected_biomass_low_total_PG = scale(expected_biomass_low)
expected_biomass_high_total_PG = scale(expected_biomass_high)

prb_total_PG = scale(mean_prb)
prb_low_total_PG = scale(mean_prb_low)
prb_high_total_PG = scale(mean_prb_high)

biomass_total = forest_biomass * forest_fraction * pixel_area * 100
biomass_total_PG = scale(forest_biomass)
biomass_low_total_PG = scale(forest_biomass_low)
biomass_high_total_PG = scale(forest_biomass_high)

dr_shift = local_disturbance_regime - natural_disturbance_regime
dr_shift_low = local_disturbance_regime_low - natural_disturbance_regime_low
dr_shift_high = local_disturbance_regime_high - natural_disturbance_regime_high

additional_carbon_storage_potential = expected_biomass_total_PG - biomass_total_PG
additional_carbon_storage_potential_low = expected_biomass_low_total_PG - biomass_low_total_PG
additional_carbon_storage_potential_high = expected_biomass_high_total_PG - biomass_high_total_PG

world['prb'] = world.apply(lambda x: aggregate_in_polygon(prb_total_PG, np.nansum, [x.geometry]),
                           axis=1)
world['dr_natural'] = world.apply(lambda x:
                                  aggregate_in_polygon(natural_disturbance_regime, np.nanmean, [x.geometry]),
                                  axis=1)
world['dr_local'] = world.apply(lambda x:
                                aggregate_in_polygon(local_disturbance_regime, np.nanmean, [x.geometry]),
                                axis=1)
world['dr_shift'] = world.apply(lambda x:
                                aggregate_in_polygon(dr_shift, np.nanmean, [x.geometry]),
                                axis=1)

world['biomass'] = world.apply(lambda x:
                               aggregate_in_polygon(biomass_total_PG, np.nansum, [x.geometry]),
                               axis=1)
world['expected_biomass'] = world.apply(lambda x:
                               aggregate_in_polygon(expected_biomass_total_PG, np.nansum, [x.geometry]),
                               axis=1)
world['clim_class_idx'] = world.apply(lambda x:
                                      aggregate_in_polygon(climate, lambda x: mode(x)[0][0], [x.geometry]),
                                      axis=1)

world['carbon_storage_potential'] = world['expected_biomass'] - world['biomass']
world = world.merge(df_clim_legend, left_on='clim_class_idx', right_index=True, how='left')
world = world.loc[world.clim_class != "Water"]


def summary(func):
    func(f"Total current biomass: {np.nansum(biomass_total_PG):.1f} Pg AGB")
    func(f"Total expected biomass: {np.nansum(expected_biomass_total_PG):.1f} Pg AGB")
    func(f"Additional carbon storage potential: "
         f"{np.nansum(additional_carbon_storage_potential):.1f} Pg AGB")
    func(f"Additional carbon storage potential: "
         f"{np.nansum(additional_carbon_storage_potential) / 2:.1f} PgC")
    func(f"Increment of: "
         f"{np.nansum(additional_carbon_storage_potential) / np.nansum(biomass_total_PG) * 100:.1f}%")
    func(f"Increment in natural areas: "
         f"{np.nansum(additional_carbon_storage_potential[intact_forest]) / np.nansum(biomass_total_PG[intact_forest]) * 100:.1f}%")
    func(f"Increment in managed areas: "
         f"{np.nansum(additional_carbon_storage_potential[managed_areas]) / np.nansum(biomass_total_PG[managed_areas]) * 100:.1f}%")
    func(f"Years of avoided carbon emissions: "
         f"{np.nansum(additional_carbon_storage_potential) / 2 / 10:.1f}")
    func("")
    func(f"Tropical rainforest maximum expected biomass: {np.nanmax(expected_biomass[climate == 1]):.1f}")
    func(f"Tropical rainforest reach up to x% higher biomass in comparison to Bastin: "
         f"{((np.nanmax(expected_biomass[climate == 1]) / 282.5) - 1) * 100:.1f}%")
    func("")
    func(f"Intact tropical rainforest mean expected biomass: "
         f"{np.nanmean(expected_biomass[np.logical_and(climate == 1, intact_forest)]):.1f}")
    func(f"Intact tropical rainforest average x% higher biomass in comparison to Bastin: "
         f"{((np.nanmean(expected_biomass[np.logical_and(climate == 1, intact_forest)]) / 282.5) - 1) * 100:.1f}%")
    func("")
    func(f"Average difference in expected and actual biomass in natural areas: "
         f"{np.nanmean((expected_biomass - forest_biomass)[intact_forest]):.1f} Mg/ha")
    mask = np.logical_and.reduce((intact_forest, ~np.isnan(expected_biomass), ~np.isnan(forest_biomass)))
    func(f"R2 expected biomass and observed biomass in natural areas: "
         f"{r2_score(expected_biomass[mask], forest_biomass[mask]):.3f}")
    mask = np.logical_and(~np.isnan(expected_biomass), ~np.isnan(forest_biomass))
    func(f"R2 expected biomass and observed biomass overall: "
         f"{r2_score(expected_biomass[mask], forest_biomass[mask]):.3f}")
    func("")
    num = 10
    func(f"Average factor increase DR in top ten counties: "
         f"{(world.sort_values('prb').tail(num).dr_local / world.sort_values('prb').tail(num).dr_natural).mean():.1f}")
    func(f"Max factor increase DR in top ten counties: "
         f"{(world.sort_values('prb').tail(num).dr_local / world.sort_values('prb').tail(num).dr_natural).max():.1f}")
    func(f"Max in: "
         f"{world.loc[(world.sort_values('prb').tail(num).dr_local / world.sort_values('prb').tail(num).dr_natural).idxmax(), 'name']}")
    func(f"Absolute maximum: "
         f"{np.nanmax(local_disturbance_regime / natural_disturbance_regime):.1f}")
    ind = np.nanargmax(local_disturbance_regime / natural_disturbance_regime)
    unraveled_ind = np.unravel_index(ind, local_disturbance_regime.shape)
    point = Point(t[unraveled_ind].x.item(), t[unraveled_ind].y.item())
    func(f"Absolute maximum in: "
         f"{world.loc[world.contains(point), 'name'].item()}")
    func("")
    val = world.sort_values("carbon_storage_potential").tail(
        5).carbon_storage_potential.sum() / world.carbon_storage_potential.sum()
    func(f"Top 5 countries have x% of additional carbon storage potential: "
         f"{val * 100:.1f}%")
    func(f"Countries: "
         f"{list(world.sort_values('carbon_storage_potential').tail(5).name)}")
    func("")
    func(f"Biomass in managed forests: "
         f"{np.nansum(biomass_total_PG[managed_areas]):.1f} Pg biomass")
    func(f"Biomass in managed forests: "
         f"{np.nansum(biomass_total_PG[managed_areas]) / 2:.1f} PgC")
    func(f"Biomass in unmanaged forests: "
         f"{np.nansum(biomass_total_PG[intact_forest]):.1f} Pg biomass")
    func(f"Fraction biomass in managed forests: "
         f"{np.nansum(biomass_total_PG[managed_areas]) / np.nansum(biomass_total_PG) * 100:.1f} %")
    func(f"Or about x times as much: "
         f"{np.nansum(biomass_total_PG[managed_areas]) / np.nansum(biomass_total_PG[intact_forest]):.1f}")
    func(f"Reached saturation in managed forests: "
         f"{np.nansum(biomass_total_PG[managed_areas]) / np.nansum(expected_biomass_total_PG[managed_areas]) * 100:.1f}%")
    func(f"Possible loss is x times larger than potential gain: "
         f"{np.nansum(biomass_total_PG[managed_areas]) / np.nansum(additional_carbon_storage_potential[managed_areas]):.1f}")
    func("")
    mask = np.logical_and(climate >= 8, climate <= 16)
    current_actual_biomass = np.nansum(scale((1 - local_disturbance_regime) * mean_prb))
    dr_natural_lower = natural_disturbance_regime.copy()
    dr_natural_lower[mask] = dr_natural_lower[mask] - 0.1
    func(f"Lower DR in temperature regions lead to additional carbon storage potential: "
         f"{(np.nansum(scale((1 - dr_natural_lower) * mean_prb)) - current_actual_biomass) / 2:.1f} PgC")
    dr_natural_higher = natural_disturbance_regime.copy()
    dr_natural_higher[mask] = dr_natural_higher[mask] + 0.1
    func(f"Higher DR in temperature regions lead to additional carbon storage potential: "
         f"{(np.nansum(scale((1 - dr_natural_higher) * mean_prb)) - current_actual_biomass) / 2:.1f} PgC")
    func("")
    func(f"Uncertainty range total biomass: "
         f"{np.nansum(biomass_low_total_PG):.1f} - "
         f"{np.nansum(biomass_high_total_PG):.1f} Pg biomass")
    func(f"Uncertainty range total expected biomass: "
         f"{np.nansum(expected_biomass_low_total_PG):.1f} - "
         f"{np.nansum(expected_biomass_high_total_PG):.1f} Pg biomass")
    func(f"Uncertainty range additional storage: "
         f"{np.nansum(additional_carbon_storage_potential_low):.1f} - "
         f"{np.nansum(additional_carbon_storage_potential_high):.1f} Pg biomass")
    func(f"Uncertainty range additional storage: "
         f"{np.nansum(additional_carbon_storage_potential_low) / 2:.1f} - "
         f"{np.nansum(additional_carbon_storage_potential_high) / 2:.1f} PgC")
    func(f"Uncertainty range additional storage: "
         f"{np.nansum(additional_carbon_storage_potential_low) / np.nansum(biomass_low_total_PG) * 100:.1f} - "
         f"{np.nansum(additional_carbon_storage_potential_high) / np.nansum(biomass_high_total_PG) * 100:.1f}%")
    natural_disturbance_regime_temperate_low = natural_disturbance_regime.copy()
    natural_disturbance_regime_temperate_low[np.logical_and(climate >= 8, climate < 17)] -= 0.1
    natural_disturbance_regime_temperate_high = natural_disturbance_regime.copy()
    natural_disturbance_regime_temperate_high[np.logical_and(climate >= 8, climate < 17)] += 0.1
    total_biomass = np.nansum(biomass_total_PG)
    func(f"Uncertainty by DR extrapolation (manual shifting in temperate zones): "
         f"{(np.nansum((1 - natural_disturbance_regime_temperate_high) * prb_total_PG) - total_biomass) / 2:.1f} - "
         f"{(np.nansum((1 - natural_disturbance_regime_temperate_low) * prb_total_PG) - total_biomass) / 2:.1f} PgC")
    mask = ~np.isnan(expected_biomass)
    average_agb = np.average(expected_biomass[mask], weights=pixel_area[mask])
    carbon_storage_potential = np.nansum(additional_carbon_storage_potential)
    func("")
    func(f"Average expected biomass: "
         f"{average_agb:.1f} Mg/ha")
    func(f"To reach the same carbon offset as by unmanaging in reforestation: "
         f"{carbon_storage_potential / (average_agb * 10**-7) / 1000000:.1f} million km2")
    func(f"Offset one year in emissions in reforestation: "
         f"{(10 * 2) / (average_agb * 10 ** -7) / 1000000:.1f} million km2")


summary(print)
with open("summary.txt", mode='w+', newline='\n') as f:
    summary(lambda x: print(x, file=f))
