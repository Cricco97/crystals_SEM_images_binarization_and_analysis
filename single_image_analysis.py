"""
Single image analysis.
"""

import os
from utils import import_label_and_store, show_or_save_fig
from feature_extraction import extract_morphological_features, extract_area_threshold
from plotting import construct_pdf_plot, construct_pdf_and_fit
from matplotlib import pyplot as plt


if __name__ == "__main__":

    nimg = 3

    cost_min_area = True
    show = False
    save_dir = 'Figures'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    bins = 60
    distr = 'lognorm'

    objects, filtered_image, settings = import_label_and_store(nimg, cost_min_area=cost_min_area, show=show, save_dir=save_dir)
    conv_factor = settings['conv_factor']

    results, areas_um2 = extract_morphological_features(nimg=nimg, objects=objects, conv_factor=conv_factor, filtered_image=filtered_image, show=show, save_dir=save_dir)
    print(results)

    thra, par = extract_area_threshold(nimg=nimg, objects=objects, conv_factor=conv_factor, show=show, save_dir=save_dir)
    print(thra, par)

    fig, ax = plt.subplots(figsize=(6, 4))
    color = 'blue'
    construct_pdf_plot(areas_um2, bins=bins, ax=ax, color=color)
    ax.set_xlabel('Area (μm²)', fontsize=12)
    ax.set_ylabel('Prob density', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle('Area Probability Density Function in log-log scale')
    if show or save_dir:
        figname = f'image_{nimg}_area_pdf'
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    fig, ax = plt.subplots(figsize=(6, 4))
    color = 'blue'
    construct_pdf_and_fit(nimg=nimg, objects=objects, conv_factor=conv_factor, bins=bins, ax=ax, color=color, distr=distr)
    ax.set_xlabel('Area (μm²)', fontsize=12)
    ax.set_ylabel('Prob density', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.suptitle('Area Probability Density Function in log-log scale')
    if show or save_dir:
        figname = f'image_{nimg}_area_pdf_{distr}_fit'
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)








