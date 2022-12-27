import warnings

import cartopy.crs as ccrs
import geomappy as mp
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.stats import mode

from plotting_helper_functions import scale, aggregate_in_polygon, load_climate_classes
from src.load_data import load_data, load_flat_file
from src.utils import BiomassParams, ModelParams, HomogeneityFactorParams

warnings.filterwarnings("ignore")

# Base model parameters for all plots. Some figures declare them specifically
biomass_params = BiomassParams()
model_params = ModelParams(biomass_params, 40, 'global', 1)
homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')

# Some standard files needed for calculations
forest_fraction = load_data('forest_fraction', model_params=model_params).values
pixel_area = load_data('pixel_area', resolution=model_params.window_size * 250).values


def get_ifl():
    IFL = gpd.read_file("data/intact_forest/ifl_2016.shp")
    IFL['values'] = 1
    return IFL


def draw_annotations(t, x, y, ax=None, overlapping_pixels=0, fontsize=10, **kwargs):
    if isinstance(ax, type(None)):
        ax = plt.gca()

    x = np.asarray(x)
    y = np.asarray(y)
    t = np.asarray(t)

    mask = np.zeros(ax.figure.canvas.get_width_height(), bool)
    plt.tight_layout()
    ax.figure.canvas.draw()

    va_positions = {'b': 'bottom', 't': 'top', 'c': 'center'}
    ha_positions = {'l': 'left', 'r': 'right', 'c': 'center'}

    indices = np.arange(len(t))

    if isinstance(fontsize, (float, int)):
        fontsize = np.repeat(fontsize, len(t))
    fontsize = np.asarray(fontsize)

    for i in indices:
        for position in ['bl', 'tl', 'tr', 'br', 'cl', 'cr', 'tc', 'bc']:
            va = va_positions[position[0]]
            ha = ha_positions[position[1]]

            a = ax.text(x=x[i], y=y[i], s=t[i], ha=ha, va=va, fontsize=fontsize[i], **kwargs)

            bbox = a.get_window_extent()
            x0 = int(bbox.x0) + overlapping_pixels
            x1 = int(np.ceil(bbox.x1)) - overlapping_pixels
            y0 = int(bbox.y0) + overlapping_pixels
            y1 = int(np.ceil(bbox.y1)) - overlapping_pixels

            s = np.s_[x0:x1 + 1, y0:y1 + 1]
            if np.any(mask[s]):
                a.set_visible(False)
            else:
                mask[s] = True
                break


def insert_hidden_ax(cax, bbox=None, color='white'):
    if bbox is None:
        bbox = inset_bbox_blank = [0.01, 0.01, 0.1, 0.7]
    hidden_ax = cax.inset_axes(bbox)
    hidden_ax.axis("off")
    hidden_ax.add_patch(Rectangle((0, 0), 1, 1, color=color))


biomass_bins = [0, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450]
gap_bins = [0, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
norm_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
fraction_bins = [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

global_ind = [-180, -65, 180, 90]
sa_ind = [-85, -19, -45, 21]
eu_ind = [-11, 30, 29, 70]


def _panel_plot(ind):
    f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 20), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.tight_layout(w_pad=7, h_pad=18)
    ax = [mp.basemap(ind, ax=cax, xticks=10, yticks=10, resolution='10m') for cax in ax.flatten()]

    mean_biomass = (
        load_data('mean_biomass', model_params=model_params)
            .rio.clip_box(*ind)
    )
    cax, cbar = mean_biomass.plot_raster(cmap="YlGn", ax=ax[0], bins=biomass_bins)
    cbar.set_label(r"Biomass [Mg ha$^{-1}$]", rotation=270, labelpad=25)

    max_biomass = (
        load_data('max_biomass', model_params=model_params)
            .rio.clip_box(*ind)
    )
    cax, cbar = max_biomass.plot_raster(cmap="YlGn", ax=ax[1], bins=biomass_bins)
    cbar.set_label(r"Biomass [Mg ha$^{-1}$]", rotation=270, labelpad=25)

    max_prb = (
        load_data('max_prb', model_params=model_params)
            .rio.clip_box(*ind)
    )
    cax, cbar = max_prb.plot_raster(cmap="YlGn", ax=ax[2], bins=biomass_bins)
    cbar.set_label(r"Biomass [Mg ha$^{-1}$]", rotation=270, labelpad=25)

    hf = (
        load_data('hf', homogeneity_factor_params=homogeneity_factor_params)
            .rio.clip_box(*ind)
    )
    cax, cbar = hf.plot_raster(cmap="YlOrBr_r", ax=ax[3], bins=fraction_bins)
    cbar.set_label(r"Fraction [-]", rotation=270, labelpad=25)

    mean_prb = (
        load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
            .rio.clip_box(*ind)
    )
    cax, cbar = mean_prb.plot_raster(cmap="YlGn", ax=ax[4], bins=biomass_bins)
    cbar.set_label(r"Biomass [Mg ha$^{-1}$]", rotation=270, labelpad=25)

    dv = 1 - (mean_biomass / mean_prb)
    cax, cbar = dv.plot_raster(cmap="YlGnBu", ax=ax[5], bins=norm_bins)
    cbar.set_label(r"Fraction [-]", rotation=270, labelpad=25)

    ax[0].set_title("Mean forest biomass", fontsize=13)
    ax[1].set_title("Maximum forest biomass", fontsize=13)
    ax[2].set_title("Maximum potential realised biomass", fontsize=13)
    ax[3].set_title("Homogeneity factor", fontsize=13)
    ax[4].set_title("Mean potential realised biomass", fontsize=13)
    ax[5].set_title("1 - (Mean forest biomass / Mean PRB)", fontsize=13)


def panel_plots():
    print("- Plotting Amazon panel")
    _panel_plot(sa_ind)
    plt.savefig("figures/panels_amazon.png", bbox_inches='tight', dpi=300)
    plt.show()

    print("- Plotting Europe panel")
    _panel_plot(eu_ind)
    plt.savefig("figures/panels_europe.png", bbox_inches='tight', dpi=300)
    plt.show()


def figure_2():
    print("- Plotting summary maps mean PRB")

    biomass_bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    gap_bins = [0, 50, 100, 150, 200, 250, 300]
    norm_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

    f, ax = plt.subplots(nrows=3, figsize=(15, 20), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.tight_layout(h_pad=0)
    ax = [mp.basemap(global_ind, ax=cax, resolution='50m', linewidth=0.7, xticks=[], yticks=[]) for cax in ax]

    inset_bbox = [0.02, 0.05, 0.01, 0.65]

    mean_prb = (
        load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
            .rio.clip_box(*global_ind)
    )
    mean_biomass = (
        load_data('mean_biomass', model_params=model_params)
            .rio.clip_box(*global_ind)
    )

    bounds = mean_prb.rio.bounds(recalc=True)
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    kwargs = dict(pad_fraction=1, aspect=30, legend_kwargs=dict(position='right'), extent=extent, fontsize=20)

    insert_hidden_ax(ax[0])
    cbar_ax = ax[0].inset_axes(inset_bbox)
    ax[0], cbar = mp.plot_raster(
        mean_prb.values,
        ax=ax[0], cmap="YlGn", bins=biomass_bins, legend_ax=cbar_ax, **kwargs)
    cbar.set_label(r"PRB [Mg ha$^{-1}$]", labelpad=15)
    ax[0].tick_params(axis='both', which='major', pad=10)

    insert_hidden_ax(ax[1])
    cbar_ax = ax[1].inset_axes(inset_bbox)
    ax[1], cbar = mp.plot_raster(
        (mean_prb - mean_biomass).values,
        ax=ax[1], cmap="pink_r", bins=gap_bins, legend_ax=cbar_ax, **kwargs)
    cbar.set_label(r"PRB gap [Mg ha$^{-1}$]", labelpad=15)
    ax[1].tick_params(axis='both', which='major', pad=10)

    insert_hidden_ax(ax[2])
    cbar_ax = ax[2].inset_axes(inset_bbox)
    ax[2], cbar = mp.plot_raster(
        (1 - (mean_biomass / mean_prb)).values,
        ax=ax[2], cmap="YlGnBu", bins=norm_bins, legend_ax=cbar_ax, **kwargs)
    cbar.set_label(r"Relative PRB gap", labelpad=15)
    ax[2].tick_params(axis='both', which='major', pad=10)

    ax[0].text(0.02, 0.96, "$\\bf{a}$", transform=ax[0].transAxes, ha='left', va='top', fontsize=22)
    ax[1].text(0.02, 0.96, "$\\bf{b}$", transform=ax[1].transAxes, ha='left', va='top', fontsize=22)
    ax[2].text(0.02, 0.96, "$\\bf{c}$", transform=ax[2].transAxes, ha='left', va='top', fontsize=22)

    plt.savefig("figures/global_maps_tight.png", bbox_inches='tight', dpi=300)
    plt.show()


def figure_3():
    print("- Plotting anthropogenic effect on disturbance regimes (scatterplot)")

    # Loading data
    natural_disturbance_regime = load_data("natural_disturbance_regime",
                                           model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params).values
    local_disturbance_regime = load_data("local_disturbance_regime",
                                         model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params).values
    forest_biomass = load_data("mean_biomass", model_params=model_params).values

    # Generate climate statistics df
    df_clim_legend = load_climate_classes()

    climate = load_flat_file("data/climate_classes/climate_classes.tif").values
    clim_dr = pd.DataFrame(columns=["clim_idx", "type", "dr"])
    for i in range(1, 31):
        mask = climate == i
        cdf_natural = pd.DataFrame(data={"clim_idx": i, "type": 'natural', "dr": natural_disturbance_regime[mask]})
        cdf_local = pd.DataFrame(data={"clim_idx": i, "type": 'local', "dr": local_disturbance_regime[mask]})
        clim_dr = pd.concat([clim_dr, cdf_natural, cdf_local])
    clim_dr = clim_dr.dropna()
    clim_dr = clim_dr.merge(df_clim_legend, left_on='clim_idx', right_index=True, how='left')

    # Generate country statistics df
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.loc[world.name == "Norway", "iso_a3"] = "NOR"
    world.loc[world.name == "France", "iso_a3"] = "FRA"
    world.loc[world.name == "N. Cyprus", "iso_a3"] = "CYP"
    world.loc[world.name == "Kosovo", "iso_a3"] = "KOS"
    world.loc[world.name == "Somaliland", "iso_a3"] = "SOM"

    world['dr_natural'] = world.apply(lambda x:
                                      aggregate_in_polygon(natural_disturbance_regime, np.nanmean, [x.geometry]),
                                      axis=1)

    world['dr_local'] = world.apply(lambda x:
                                    aggregate_in_polygon(local_disturbance_regime, np.nanmean, [x.geometry]),
                                    axis=1)

    biomass_total = scale(forest_biomass)
    world['biomass'] = world.apply(lambda x:
                                   aggregate_in_polygon(biomass_total, np.nansum, [x.geometry]),
                                   axis=1)

    world['clim_class_idx'] = world.apply(lambda x:
                                          aggregate_in_polygon(climate, lambda x: mode(x)[0][0], [x.geometry]),
                                          axis=1)

    world = world.merge(df_clim_legend, left_on='clim_class_idx', right_index=True, how='left')
    world = world.loc[world.clim_class != "Water"]
    plot_world = world.dropna().sort_values("biomass", ascending=False)[:-10]

    # Scatterplot
    calc_dot_size = lambda x: (np.sqrt(x) + 1) * 50
    calc_annot_size = lambda x: (np.sqrt(x) + 10) * 1

    f, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(plot_world.dr_natural, plot_world.dr_local,
                    s=calc_dot_size(plot_world.biomass),
                    hue=plot_world.clim_class_major,
                    hue_order=["A", "B", "C", "D", "E"],
                    palette=["#2CA02C", "#FF7F0F", "#D62728", "#1F77B4", "#9467BD"],
                    edgecolor="grey", alpha=0.35, ax=ax)
    ax.set_xlim(0.29, 0.76)
    ax.set_ylim(0.29, 0.94)
    ax.plot([0, 1], [0, 1], color='grey')
    ax.plot([0, 1], [0.1, 1.1], color='grey', linestyle=":")
    ax.plot([0, 1], [0.2, 1.2], color='grey', linestyle=":")
    draw_annotations(plot_world.iso_a3, plot_world.dr_natural, plot_world.dr_local,
                     fontsize=calc_annot_size(plot_world.biomass), ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Natural disturbance regime [-]", fontsize=14)
    ax.set_ylabel("Local disturbance regime [-]", fontsize=14)
    ax.text(0.025, 0.96, "$\\bf{a}$", transform=ax.transAxes, ha='left', va='top', fontsize=18)
    legend = ax.legend(
        handles=mp.colors.legend_patches(["#2CA02C", "#FF7F0F", "#D62728", "#1F77B4", "#9467BD"],
                                         ["Tropical (A)", "Dry (B)", "Temperate (C)", "Continental (D)", "Boreal (E)"],
                                         type="", marker="o", edgecolor="none", alpha=0.6),
        title="Climate zone", loc='upper right', fontsize=12
    )
    legend._legend_box.align = "left"

    l = []
    for x in [1, 10, 50]:
        l.append(plt.scatter([], [], marker='o', edgecolor="grey", facecolor="none", s=calc_dot_size(x),
                             label=f"{x:.0f} Pg"))
    ax.legend(handles=l, loc='lower center', title="Total observed biomass", ncol=3,
              labelspacing=1.05,
              borderpad=1.2)
    ax.add_artist(legend)
    ax.text(x=0.31, y=0.34, s="1:1", ha='center', fontsize=14)
    ax.text(x=0.31, y=0.44, s="+10%", ha='center', fontsize=14)
    ax.text(x=0.31, y=0.54, s="+20%", ha='center', fontsize=14)

    # Inset in scatterplot
    inset_ax = ax.inset_axes([0.72, 0.06, 0.25, 0.32])
    sns.barplot(data=clim_dr, x='clim_class_major', y='dr', hue='type', palette=['darkseagreen', 'sandybrown'],
                ax=inset_ax,
                errwidth=1, ci='sd')
    inset_ax.legend(bbox_to_anchor=(0.01, 0.94), loc='lower left', fontsize=12)
    inset_ax.get_legend().set_title("")
    inset_ax.get_legend().get_texts()[0].set_text("Intact forest landscapes")
    inset_ax.get_legend().get_texts()[1].set_text("Human altered landscapes")
    inset_ax.set_ylim(0, 1)
    inset_ax.set_xlabel("")
    inset_ax.text(0.07, 0.95, "$\\bf{b}$", transform=inset_ax.transAxes, ha='left', va='top', fontsize=18)
    inset_ax.set_ylabel("Disturbance regime [-]", fontsize=14)
    inset_ax.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(ax=inset_ax)

    plt.savefig("figures/shift_disturbance_regime_scatterplot.png", dpi=300, bbox_inches='tight')
    plt.show()


def figure_4():
    print("- Plotting additional global carbon storage potential")
    f, ax = plt.subplots(nrows=2, figsize=(15, 15), gridspec_kw={'height_ratios': [2, 1]})
    plt.tight_layout(h_pad=25)
    ax[0] = mp.basemap(global_ind, coastline_linewidth=0.5, resolution='50m', ax=ax[0], fontsize=14, padding=25,
                       xticks=[], yticks=[])

    mean_prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    natural_disturbance_regime = load_data('natural_disturbance_regime', model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params)
    local_disturbance_regime = load_data('local_disturbance_regime', model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params)

    # Map
    bounds = mean_prb.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    biomass_storage_potential = ((local_disturbance_regime - natural_disturbance_regime) * mean_prb)
    insert_hidden_ax(ax[0])
    cbar_ax = ax[0].inset_axes([0.02, 0.05, 0.01, 0.6])
    ax[0], cbar = mp.plot_raster(
        biomass_storage_potential.values,
        ax=ax[0], cmap="PuOr_r", vmin=-100, vmax=100, fontsize=14, pad_fraction=1, aspect=30, extent=extent,
        legend_ax=cbar_ax, legend_kwargs=dict(position='right'))
    ax[0].text(0.02, 0.96, "$\\bf{a}$", transform=ax[0].transAxes, ha='left', va='top', fontsize=18)
    cbar.set_label(r"Additional carbon storage potential [Mg ha$^{-1}$]")

    # Barplot
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.loc[world.name == "Norway", "iso_a3"] = "NOR"
    world.loc[world.name == "France", "iso_a3"] = "FRA"
    world.loc[world.name == "N. Cyprus", "iso_a3"] = "CYP"
    world.loc[world.name == "Kosovo", "iso_a3"] = "KOS"
    world.loc[world.name == "Somaliland", "iso_a3"] = "SOM"

    world['biomass_storage_potential'] = (
        world.apply(lambda x: aggregate_in_polygon(scale(biomass_storage_potential.values),
                                                   func=np.nansum, polygon=[x.geometry]), axis=1))

    norm = matplotlib.colors.Normalize(vmin=-8, vmax=8)
    cmap = plt.get_cmap("PuOr_r")
    cdf = (
        world.sort_values('biomass_storage_potential', ascending=False)
            .loc[world.biomass_storage_potential.abs() > 0.5]
    )
    cdf['colors'] = cdf.apply(lambda x: cmap(norm(x.biomass_storage_potential)), axis=1)
    cdf['extended_name'] = cdf.name + " (" + cdf.iso_a3 + ")"
    sns.barplot(data=cdf, y='biomass_storage_potential', x='extended_name', ax=ax[1], palette=cdf.colors)
    ax[1].set_xlabel("")
    ax[1].axhline(0, color='grey', linestyle="--")
    ax[1].set_ylabel("Additional carbon storage potential [Pg biomass]", fontsize=14)
    ax[1].text(0.02, 0.96, "$\\bf{b}$", transform=ax[1].transAxes, ha='left', va='top', fontsize=18)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    ax[1].tick_params(axis='x', rotation=90)
    ax[1].set_ylim(ax[1].get_ylim()[0], 13)

    plt.savefig("figures/addition_carbon_storage_potential.png", dpi=280, bbox_inches='tight')
    plt.show()


def figure_4_2():
    print("- Plotting additional global carbon storage potential (simplified)")

    f, ax = plt.subplots(figsize=(15, 10))
    ax = mp.basemap(extent=[-180, -70, 180, 90], coastline_linewidth=0.5, resolution='50m', ax=ax, fontsize=14,
                    padding=25,
                    xticks=[], yticks=[])

    mean_prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    natural_disturbance_regime = load_data('natural_disturbance_regime', model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params)
    local_disturbance_regime = load_data('local_disturbance_regime', model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params)

    # Map
    bounds = mean_prb.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    biomass_storage_potential = ((local_disturbance_regime - natural_disturbance_regime) * mean_prb)
    insert_hidden_ax(ax, [0.01, 0.01, 0.15, 0.65])
    insert_hidden_ax(ax, [0.001, 0.001, 0.99, 0.05])
    ax, cbar = mp.plot_raster(
        biomass_storage_potential.values,
        ax=ax, cmap="PuOr_r", vmin=-100, vmax=100, fontsize=14, pad_fraction=1, aspect=50, extent=extent,
        legend_kwargs=dict(position='right', shrink=0.6))
    ax.text(0.02, 0.98, "$\\bf{a}$", transform=ax.transAxes, ha='left', va='top', fontsize=18)
    cbar.set_label(r"CSP [Mg ha$^{-1}$]")

    # Barplot
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.loc[world.name == "Norway", "iso_a3"] = "NOR"
    world.loc[world.name == "France", "iso_a3"] = "FRA"
    world.loc[world.name == "N. Cyprus", "iso_a3"] = "CYP"
    world.loc[world.name == "Kosovo", "iso_a3"] = "KOS"
    world.loc[world.name == "Somaliland", "iso_a3"] = "SOM"

    world['biomass_storage_potential'] = (
        world.apply(lambda x: aggregate_in_polygon(scale(biomass_storage_potential.values),
                                                   func=np.nansum, polygon=[x.geometry]), axis=1))

    inset_ax = ax.inset_axes([0, 0, 1, 0.36])
    inset_ax.set_facecolor("None")
    sns.despine(ax=inset_ax, top=True, right=True)

    norm = matplotlib.colors.Normalize(vmin=-8, vmax=8)
    cmap = plt.get_cmap("PuOr_r")
    cdf = (
        world.sort_values('biomass_storage_potential', ascending=False)
            .loc[world.biomass_storage_potential.abs() > 0.7]
    )
    cdf['colors'] = cdf.apply(lambda x: cmap(norm(x.biomass_storage_potential)), axis=1)
    cdf['extended_name'] = cdf.name + " (" + cdf.iso_a3 + ")"
    sns.barplot(data=cdf, y='biomass_storage_potential', x='extended_name', ax=inset_ax, palette=cdf.colors)
    inset_ax.set_xlabel("")
    inset_ax.axhline(0, color='grey', linestyle="--")
    inset_ax.set_ylabel("CSP [Pg biomass]", fontsize=14)
    inset_ax.text(0.04, 0.94, "$\\bf{b}$", transform=inset_ax.transAxes, ha='left', va='top', fontsize=18)
    inset_ax.tick_params(axis='both', which='major', labelsize=14)
    inset_ax.tick_params(axis='x', rotation=90)
    xlim = inset_ax.get_xlim()
    inset_ax.set_xlim(xlim[0] - 0.2, xlim[1] + 0.2)

    plt.savefig("figures/addition_carbon_storage_potential_simplified.png", dpi=280, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_1():
    print("- Plotting uncertainty experiment 1: percentual increase biomass")

    mean_prb_dict = {}
    expected_biomass_dict = {}

    for percent_increase in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        biomass_params = BiomassParams(scale=1 + percent_increase / 100)
        model_params = ModelParams(biomass_params, 40, 'global', 1)

        mean_prb = load_data("mean_prb", model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
        expected_biomass = load_data("expected_biomass", model_params=model_params,
                                     homogeneity_factor_params=homogeneity_factor_params)

        mean_prb_dict[percent_increase] = scale(mean_prb).sum().values.item()
        expected_biomass_dict[percent_increase] = scale(expected_biomass).sum().values.item()

    df = pd.DataFrame.from_dict(expected_biomass_dict, orient='index')

    f, ax = plt.subplots(figsize=(8, 8))
    ax.plot([-1, 23], [-1, 23], color='lightgrey', alpha=1, zorder=-1)
    sns.scatterplot(data=df, x=df.index, y=(df[0] / df[0][0] - 1) * 100, color='darkblue', ax=ax, marker='x',
                    linewidth=2)
    ax.text(x=21, y=23, s='1:1')
    ax.set_xlabel("Biomass increase [%]")
    ax.set_ylabel("Biomass carrying capacity increase [%]")
    plt.savefig("figures/uncertainty_analysis_1_simple.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_2():
    print("- Plotting uncertainty experiment 2: Monte Carlo")

    biomass_params = BiomassParams()
    model_params = ModelParams(biomass_params, 40, 'global', 1)
    eb = load_data('expected_biomass', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)

    eb_ensemble = np.full(eb.shape + (50,), np.nan)
    for i in range(50):
        biomass_params = BiomassParams(noise=True, seed=i)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        eb_ensemble[:, :, i] = load_data('expected_biomass', model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params).values

    std = eb_ensemble.std(axis=-1)

    bounds = eb.rio.bounds()
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster(std / eb.values, ax=ax, pad_fraction=3, aspect=30, vmax=1,
                              legend_kwargs=dict(position='bottom', shrink=0.8), cmap="Oranges",
                              bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1], extent=extent)
    cbar.set_label("Coefficient of variation of BCC predictions [-]")
    plt.savefig("figures/uncertainty_analysis_2.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_3():
    print("- Plotting uncertainty experiment 3: different quantiles")

    increment_dict = {}

    for q in [0.9, 0.95, 0.99, 1]:
        biomass_params = BiomassParams()
        model_params = ModelParams(biomass_params, 40, 'global', q)
        prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
        eb = load_data('expected_biomass', model_params=model_params,
                       homogeneity_factor_params=homogeneity_factor_params)
        increment_dict[q] = (
            scale(prb).sum().values.item(),
            scale(eb).sum().values.item()
        )

    df = pd.DataFrame.from_dict(increment_dict, orient='index', columns=['prb', 'eb'])
    df = (df / df.iloc[-1] - 1) * 100

    f, ax = plt.subplots(ncols=2, figsize=(12, 8))
    sns.barplot(data=df, x=df.index, y=df.prb, color='lightgrey', ax=ax[0])
    sns.barplot(data=df, x=df.index, y=df.eb, color='lightgrey', ax=ax[1])

    def _decorate_subplot(i, title, col, prec):
        ax[i].plot([-0.5, 3.5], [0, 0], color='darkblue', alpha=0.6, linewidth=1, linestyle='--')
        ax[i].set_ylim(-23, 10)
        ax[i].set_xlabel("Quantile in framework [-]")
        ax[i].set_ylabel(title)
        mean_changes = [df.loc[q, col].item() for q in increment_dict]
        for j in range(len(mean_changes)):
            ax[i].text(x=j, y=1, s=f"{mean_changes[j]:+.{prec}f}%", ha='center', va='top', color='darkblue', alpha=0.6,
                       fontsize=10)

    _decorate_subplot(0, "Mean PRB change [%]", 'prb', 0)
    _decorate_subplot(1, "Mean BCC change [%]", 'eb', 1)

    plt.savefig("figures/uncertainty_analysis_3_simple.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_4():
    print("- Plotting uncertainty experiment 4: increase biomass with steps in standard deviation")

    prb_increment_dict = {}
    eb_increment_dict = {}

    biomass_params = BiomassParams()
    model_params = ModelParams(biomass_params, 40, 'global', 1)
    prb_base = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    eb_base = load_data('expected_biomass', model_params=model_params,
                        homogeneity_factor_params=homogeneity_factor_params)

    for s in [0, 0.5, 1, 1.5, 2, 2.5]:
        biomass_params = BiomassParams(error_scale=s)
        model_params = ModelParams(biomass_params, 40, 'global', 1)
        prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
        eb = load_data('expected_biomass', model_params=model_params,
                       homogeneity_factor_params=homogeneity_factor_params)
        mask = ~np.isnan(prb.values)
        prb_increment_dict[s] = ((prb.values[mask] / prb_base.values[mask]) - 1) * 100
        eb_increment_dict[s] = ((eb.values[mask] / eb_base.values[mask]) - 1) * 100

    df_prb = pd.DataFrame(prb_increment_dict).melt()
    df_eb = pd.DataFrame(eb_increment_dict).melt()

    f, ax = plt.subplots(ncols=2, figsize=(12, 8))
    sns.boxplot(data=df_prb, x='variable', y='value', color='lightgrey',
                ax=ax[0], showfliers=False)
    sns.boxplot(data=df_eb, x='variable', y='value', color='lightgrey',
                ax=ax[1], showfliers=False)

    def _decorate_subplot(i, title, d):
        ax[i].set_xlabel("Biomass standard deviation addition [-]")
        ax[i].set_ylabel(title)
        for j, k in enumerate(d):
            ax[i].text(x=j, y=-30, s=f"{d[k].mean():+.0f}%", ha='center', color='darkblue', alpha=0.6)
        ax[i].set_ylim(-40, 260)

    _decorate_subplot(0, "Mean PRB increase [%]", prb_increment_dict)
    _decorate_subplot(1, "Mean BCC increase [%]", eb_increment_dict)

    plt.savefig("figures/uncertainty_analysis_4_simple.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_5():
    print("- Plotting uncertainty experiment 5: different resolutions")

    prb_increment_dict = {}
    eb_increment_dict = {}

    biomass_params = BiomassParams()
    model_params = ModelParams(biomass_params, 40, 'global', 1)
    homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')
    prb_base = (
        scale(
            load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
        ).sum().values.item())
    eb_base = (
        scale(
            load_data('expected_biomass', model_params=model_params,
                      homogeneity_factor_params=homogeneity_factor_params)
        ).sum().values.item())

    for ws in [40, 60, 80, 100]:
        biomass_params = BiomassParams()
        model_params = ModelParams(biomass_params, ws, 'global', 1)
        homogeneity_factor_params = HomogeneityFactorParams(biomass_params, ws, 'global')

        forest_fraction = load_data("forest_fraction", model_params=model_params)
        pixel_area = load_data("pixel_area", resolution=ws * 250)

        prb = load_data('mean_prb', model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
        eb = load_data('expected_biomass', model_params=model_params,
                       homogeneity_factor_params=homogeneity_factor_params)

        prb_increment_dict[ws] = ((prb * forest_fraction * pixel_area * 100 * 10 ** -9).sum().item() / prb_base - 1) * 100
        eb_increment_dict[ws] = ((eb * forest_fraction * pixel_area * 100 * 10 ** -9).sum().item() / eb_base - 1) * 100

    df_prb = pd.DataFrame.from_dict(prb_increment_dict, orient='index')
    df_eb = pd.DataFrame.from_dict(eb_increment_dict, orient='index')

    f, ax = plt.subplots(ncols=2, figsize=(12, 8))
    sns.barplot(data=df_prb, x=df_prb.index, y=df_prb[0], color='lightgrey', ax=ax[0])
    sns.barplot(data=df_eb, x=df_eb.index, y=df_eb[0], color='lightgrey', ax=ax[1])

    def _decorate_subplot(i, title, d):
        ax[i].plot([-0.5, 3.5], [0, 0], color='darkblue', alpha=0.6, linewidth=1, linestyle='--')
        ax[i].set_ylim(-10, 2)
        ax[i].set_xlabel("Resolution [arc second]")
        ax[i].set_xticklabels([300, 450, 600, 750])
        ax[i].set_ylabel(title)
        mean_changes = [d[ws] for ws in d]
        for j in range(len(mean_changes)):
            ax[i].text(x=j, y=0.4, s=f"{mean_changes[j]:+.1f}%", ha='center', va='top', color='darkblue', alpha=0.6,
                       fontsize=10)

    _decorate_subplot(0, "Mean PRB change [%]", prb_increment_dict)
    _decorate_subplot(1, "Mean BCC change [%]", eb_increment_dict)

    plt.savefig("figures/uncertainty_analysis_5_simple.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_uncertainty_experiment_6():
    print("- Plotting uncertainty experiment 6: SREX units")

    biomass_params = BiomassParams()
    model_params = ModelParams(biomass_params, 40, 'global', 1)
    homogeneity_factor_params = HomogeneityFactorParams(biomass_params, 40, 'global')
    prb = load_data("mean_prb", model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)
    eb = load_data("expected_biomass", model_params=model_params, homogeneity_factor_params=homogeneity_factor_params)

    model_params_srex = ModelParams(biomass_params, 40, 'global', 1, True)
    prb_srex = load_data("mean_prb", model_params=model_params_srex,
                         homogeneity_factor_params=homogeneity_factor_params)
    eb_srex = load_data("expected_biomass", model_params=model_params_srex,
                        homogeneity_factor_params=homogeneity_factor_params)

    bounds = prb.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster((prb_srex / prb - 1).values * 100, vmin=-50, vmax=50, cmap="PRGn",
                              ax=ax, pad_fraction=3, aspect=30, extent=extent,
                              legend_kwargs=dict(position='bottom', shrink=0.8, extend='both'))
    cbar.set_label("Mean PRB shift [%]")
    plt.savefig("figures/uncertainty_analysis_6.png", dpi=300, bbox_inches='tight')
    plt.show()

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster((eb_srex / eb - 1).values * 100, vmin=-50, vmax=50, cmap="PRGn",
                              ax=ax, pad_fraction=3, aspect=30, extent=extent,
                              legend_kwargs=dict(position='bottom', shrink=0.8, extend='both'))
    cbar.set_label("Mean PRB shift [%]")
    plt.savefig("figures/uncertainty_analysis_6_eb.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_natural_disturbance_regime(**kwargs):
    print("- Plotting map natural disturbance regime")
    natural_disturbance_regime = load_data("natural_disturbance_regime",
                                           model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params)
    bounds = natural_disturbance_regime.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    x, cbar = mp.plot_raster(natural_disturbance_regime.values, bins=np.linspace(0, 1, 11), cmap="YlGnBu", ax=ax,
                             extent=extent, **kwargs)
    cbar.set_label(r"Disturbance regime [-]")
    plt.savefig("figures/map_natural_dr.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_local_disturbance_regime(**kwargs):
    print("- Plotting map local disturbance regime")
    local_disturbance_regime = load_data("local_disturbance_regime",
                                         model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params)
    bounds = local_disturbance_regime.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster(local_disturbance_regime.values, bins=np.linspace(0, 1, 11), cmap="YlGnBu", ax=ax,
                              extent=extent, **kwargs)
    cbar.set_label(r"Disturbance regime [-]")
    plt.savefig("figures/map_local_dr.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_shift_disturbance_regime(**kwargs):
    print("- Plotting map shift disturbance regime")
    natural_disturbance_regime = load_data("natural_disturbance_regime",
                                           model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params)
    local_disturbance_regime = load_data("local_disturbance_regime",
                                         model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params)
    bounds = local_disturbance_regime.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster((local_disturbance_regime - natural_disturbance_regime).values, cmap="PuOr_r", vmin=-0.7,
                              vmax=0.7, ax=ax, extent=extent, **kwargs)
    get_ifl().plot_shapes(ax=ax, hatch="...", facecolor="none", linewidth=0)
    cbar.set_label(r"Distance natural disturbance regime [-]")
    plt.savefig("figures/map_shift_dr.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_shift_disturbance_regime_biomass(**kwargs):
    print("- Plotting map shift disturbance regime in biomass terms")
    natural_disturbance_regime = load_data("natural_disturbance_regime",
                                           model_params=model_params,
                                           homogeneity_factor_params=homogeneity_factor_params)
    local_disturbance_regime = load_data("local_disturbance_regime",
                                         model_params=model_params,
                                         homogeneity_factor_params=homogeneity_factor_params)
    prb = load_data("mean_prb",
                    model_params=model_params,
                    homogeneity_factor_params=homogeneity_factor_params)

    bounds = local_disturbance_regime.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster(((local_disturbance_regime - natural_disturbance_regime) * prb).values, ax=ax,
                              cmap="PuOr_r", vmin=-150, vmax=150, extent=extent, **kwargs)
    get_ifl().plot_shapes(ax=ax, facecolor="none", linewidth=0, hatch='...')
    cbar.set_label(r"Distance natural disturbance regime [Mg ha$^{-1}$]")
    plt.savefig("figures/map_shift_dr_biomass.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_climate_distance_to_natural(**kwargs):
    print("- Plotting climate distance to natural forests")
    # ax = mp.basemap(M_forest_biomass.bounds, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    # ax, cbar = mp.plot_raster(natural_dist, ax=ax, bins=[0, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], cmap='pink_r', **kwargs)
    # cbar.set_label(r"Climate coordinate distance [-]")
    # plt.savefig("figures/distance_dr_predictions.png", dpi=300, bbox_inches='tight')
    # plt.show()
    pass


def plot_expected_biomass(**kwargs):
    print("- Plotting map expected biomass")
    expected_biomass = load_data("expected_biomass",
                                 model_params=model_params,
                                 homogeneity_factor_params=homogeneity_factor_params)
    bounds = expected_biomass.rio.bounds(recalc=True)
    extent = bounds[0], bounds[2], bounds[1], bounds[3]

    ax = mp.basemap(global_ind, figsize=(20, 20), xlines=[], ylines=[], linewidth=0.5, resolution='50m')
    ax, cbar = mp.plot_raster(expected_biomass.values, ax=ax, vmin=0, vmax=300, cmap='Greens',
                              bins=[0, 20, 50, 100, 150, 200, 250, 300], extent=extent, **kwargs)
    cbar.set_label(r"Biomass carrying capacity [Mg/ha]")
    plt.savefig("figures/expected_biomass.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_r2_expected_biomass():
    print("- Plotting R2 expected biomass vs observed biomass")

    expected_biomass = load_data('expected_biomass', model_params=model_params,
                                 homogeneity_factor_params=homogeneity_factor_params).values
    forest_biomass = load_data("mean_biomass", model_params=model_params).values
    intact_forest = load_data('intact_forest', model_params=model_params).values

    natural_forest_mask = np.logical_and.reduce((intact_forest, ~np.isnan(expected_biomass), ~np.isnan(forest_biomass)))
    forest_mask = np.logical_and(~np.isnan(expected_biomass), ~np.isnan(forest_biomass))

    f, ax = plt.subplots(ncols=2, figsize=(10, 5))
    sns.scatterplot(forest_biomass[natural_forest_mask], expected_biomass[natural_forest_mask], ax=ax[0])
    sns.scatterplot(forest_biomass[forest_mask], expected_biomass[forest_mask], ax=ax[1])

    def _decorate_ax(i):
        ax[i].plot([0, 450], [0, 450])
        ax[i].text(400, 445, '1:1')
        ax[i].set_xlabel(r"Observed biomass [Mg ha$^{-1}$]")
        ax[i].set_ylabel(r"Modelled biomass carrying capacity [Mg ha$^{-1}$]")
        ax[i].text(10, 430, r"$\bf{" + chr(i + 97) + "}$", fontsize=14)

    _decorate_ax(0)
    _decorate_ax(1)

    plt.tight_layout()
    plt.savefig("figures/r2_expected_biomass.png", bbox_inches='tight', dpi=300)
    plt.show()


def main():
    # main plots
    figure_2()  # global maps PRB compared to observed biomass
    figure_3()  # scatterplot countries
    figure_4()  # global map carbon storage potential and country stats
    figure_4_2()  # same but condensed

    # uncertainty plots
    plot_uncertainty_experiment_1()
    plot_uncertainty_experiment_2()
    plot_uncertainty_experiment_3()
    plot_uncertainty_experiment_4()
    plot_uncertainty_experiment_5()
    plot_uncertainty_experiment_6()

    # extra plots
    panel_plots()
    plot_r2_expected_biomass()

    kwargs = dict(pad_fraction=3, aspect=30, legend_kwargs=dict(position='bottom', shrink=0.8))
    plot_natural_disturbance_regime(**kwargs)
    plot_local_disturbance_regime(**kwargs)
    plot_shift_disturbance_regime(**kwargs)
    plot_shift_disturbance_regime_biomass(**kwargs)
    plot_climate_distance_to_natural(**kwargs)
    plot_expected_biomass(**kwargs)


if __name__ == '__main__':
    main()
