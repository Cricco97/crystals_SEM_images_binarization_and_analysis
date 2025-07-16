"""
Binary images feature extraction pipeline.
"""

import numpy as np
from utils import compute_surface_coverage, round_value_by_error, round_value, fractal_dimension_measure, find_threshold_ks
from statsmodels import robust


def extract_morphological_features(nimg, objects, conv_factor, filtered_image, show=False, save_dir=None):

    n_struct = len(objects['area'])
    areas_pixel = np.array(objects['area'])
    perimeters_pixel = np.array(objects['perimeter'])
    areas_um2 = areas_pixel / (conv_factor ** 2)
    perimeters_um2 = perimeters_pixel / conv_factor

    median_area = np.median(areas_um2)
    mad_area = robust.mad(areas_um2) / 2
    median_area, mad_area = round_value_by_error(median_area, mad_area)
    mean_area = np.mean(areas_um2)
    mean_area = round_value(mean_area)

    coverage = compute_surface_coverage(filtered_image)
    coverage = round(coverage, 1)

    fract_dim, fract_dim_err = fractal_dimension_measure(nimg=nimg, areas=areas_um2, perimeters=perimeters_um2, show=show, save_dir=save_dir)
    fract_dim, fract_dim_err = round_value_by_error(fract_dim, fract_dim_err)

    total_image_area = filtered_image.size / (conv_factor ** 2)
    density = n_struct / total_image_area
    density = round(density, 2)

    results = {
        'n_structures': n_struct,
        'median_area': median_area,
        'mad_area': mad_area,
        'mean_area': mean_area,
        'surface_coverage': coverage,
        'fractal_dimension': fract_dim,
        'fractal_dimension_err': fract_dim_err,
        'structure_density':density
    }

    return results, areas_um2


def extract_area_threshold(nimg, objects, conv_factor, show=False, save_dir=None, maximum=200):

    areas_pixel = np.array(objects['area'])
    areas_um2 = areas_pixel / (conv_factor ** 2)

    thra, par = find_threshold_ks(nimg, areas_um2, maximum, conv_factor, 'lognorm', show=show, save_dir=save_dir)

    return thra, par