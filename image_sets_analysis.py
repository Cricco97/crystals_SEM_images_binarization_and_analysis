"""
Image sets analysis.
"""

import os
from statsmodels import robust
from utils import import_label_and_store, show_or_save_fig, round_value_by_error
from feature_extraction import extract_morphological_features, extract_area_threshold
from plotting import construct_pdf_plot
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


if __name__ == "__main__":

    images_per_substrate = [
        [13, 31, 32, 10, 11, 3, 4],
        [12, 19, 20, 28, 8, 16, 26, 27, 6, 15, 23, 24, 25]
    ]

    substrates = ['Glass', 'Silicon']

    cost_min_area = False
    show = False
    figures_dir = 'Figures'
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
    files_dir = 'Files'
    if files_dir:
        os.makedirs(files_dir, exist_ok=True)

    if cost_min_area:
        file_path = f"measurements_per_substrate_before_threshold.csv"
    else:
        file_path = f"measurements_per_substrate_after_threshold.csv"

    columns_measurements_df = ["SUBSTRATE",
                               "N STRUCT", "N NW",
                               "MEDIAN AREA", "MAD AREA",
                               "MEAN COV SURF", "ERR COV SURF",
                               "MEAN FRAC DIM", "ERR FRAC DIM",
                               "MEAN DENSITY", "ERR DENSITY"
                               ]

    measurements_df = pd.DataFrame(columns=columns_measurements_df)

    bins = 300
    distr = 'lognorm'
    colors = ['blue', 'green']

    columns_thr_area_df = ["IMG", "THRA", "PARAM1", "PARAM2", "PARAM3"]

    thr_area_df = pd.DataFrame(columns=columns_thr_area_df)

    fig, ax = plt.subplots(figsize=(6, 4))

    for sub in range(2):

        substrate = substrates[sub]

        print(f'{substrate}')

        all_areas_um2 = []
        surface_coverages = []
        fractal_dims = []
        densities = []

        n_total_structures = 0
        n_total_nanowires = 0

        for nimg in images_per_substrate[sub]:

            objects, filtered_image, settings = import_label_and_store(nimg, cost_min_area=cost_min_area, show=show, save_dir=None)
            conv_factor = settings['conv_factor']

            results, areas_um2 = extract_morphological_features(nimg=nimg, objects=objects, conv_factor=conv_factor, filtered_image=filtered_image, show=show, save_dir=None)

            all_areas_um2.extend(areas_um2)
            surface_coverages.append(results['surface_coverage'])
            fractal_dims.append(results['fractal_dimension'])
            densities.append(results['structure_density'])

            n_total_structures += results['n_structures']
            n_total_nanowires += settings["n of nanowires"]

            thra, par = extract_area_threshold(nimg=nimg, objects=objects, conv_factor=conv_factor, show=show, save_dir=None)
            par1, par2, par3 = par

            row = pd.DataFrame([{
                "IMG": nimg,
                "THRA": thra,
                "PARAM1": par1,
                "PARAM2": par2,
                "PARAM3": par3
            }])

            thr_area_df = pd.concat([thr_area_df, row], ignore_index=True)

        median_area = np.median(all_areas_um2)
        mad_area = robust.mad(all_areas_um2) / 2
        median_area, mad_area = round_value_by_error(median_area, mad_area)

        mean_cov = np.mean(surface_coverages)
        err_cov = np.std(surface_coverages)
        mean_cov, err_cov = round_value_by_error(mean_cov, err_cov)

        mean_fd = np.mean(fractal_dims)
        err_fd = np.std(fractal_dims)
        mean_fd, err_fd = round_value_by_error(mean_fd, err_fd)

        mean_density = np.mean(densities)
        err_density = np.std(densities)
        mean_density, err_density = round_value_by_error(mean_density, err_density)

        row = pd.DataFrame([{
            "SUBSTRATE": substrate,
            "N STRUCT": n_total_structures,
            "N NW": n_total_nanowires,
            "MEDIAN AREA": median_area,
            "MAD AREA": mad_area,
            "MEAN COV SURF": mean_cov,
            "ERR COV SURF": err_cov,
            "MEAN FRAC DIM": mean_fd,
            "ERR FRAC DIM": err_fd,
            "MEAN DENSITY": mean_density,
            "ERR DENSITY": err_density
        }])

        measurements_df = pd.concat([measurements_df, row], ignore_index=True)

        color = colors[sub]
        construct_pdf_plot(all_areas_um2, bins=bins, ax=ax, color=color)

    if files_dir:
        thr_area_df.to_csv(os.path.join(files_dir, 'area_threshold_parameters.csv'), sep=';', index=False, float_format='%.6f', decimal='.')
        measurements_df.to_csv(os.path.join(files_dir, file_path), sep=';', index=False, float_format='%.6f', decimal='.')

    ax.set_xlabel('Area (μm²)', fontsize=12)
    ax.set_ylabel('Prob density', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle('Area Probability Density Function in log-log scale')
    if show or figures_dir:
        figname = f'area_pdf_combined_glass_and_silicon'
        show_or_save_fig(fig=fig, show=show, save_dir=figures_dir, figname=figname)






