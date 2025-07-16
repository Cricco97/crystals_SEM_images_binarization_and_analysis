'''
Plotting util functions
'''

from utils import round_value, round_value_by_error
import numpy as np
from statsmodels import robust
from scipy.stats import lognorm, pareto


def construct_pdf_plot(areas_um2, bins=60, ax=None, color='blue'):

    median_area = np.median(areas_um2)
    mad_area = robust.mad(areas_um2) / 2
    median_area, mad_area = round_value_by_error(median_area, mad_area)
    mean_area = np.mean(areas_um2)
    mean_area = round_value(mean_area)

    counts, bin_edges = np.histogram(areas_um2, bins=bins, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.plot(bin_centres, counts, '.', color=color)
    ax.axvline(x=median_area, color=color, linestyle='-', linewidth=1)
    ax.axvline(x=mean_area, color=color, linestyle='--', linewidth=1)
    ax.axvspan(median_area - mad_area, median_area + mad_area, color=color, alpha=0.1)
    ax.set_xlabel('Area (μm²)', fontsize=24)
    ax.set_ylabel('Prob density', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)

def construct_pdf_and_fit(nimg, objects, conv_factor, bins=60, ax=None, color='blue', distr='lognorm'):

    areas_pixel = np.array(objects['area'])
    areas_um2 = areas_pixel / (conv_factor ** 2)

    counts, bin_edges = np.histogram(areas_um2, bins=bins, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    x_pdf = np.linspace(min(areas_um2), max(areas_um2), 1000)

    if distr == 'lognorm':
        params = lognorm.fit(areas_um2)
        theoretical_pdf = lognorm.pdf(x_pdf, params[0], params[1], params[2])
    elif distr == 'pareto':
        params = pareto.fit(areas_um2)
        theoretical_pdf = pareto.pdf(x_pdf, params[0], params[1], params[2])
    else:
        print('define the distribution inside the function')
        return

    theoretical_pdf = theoretical_pdf * len(areas_um2) * np.diff(bin_edges)[0]

    ax.plot(bin_centres, counts, '.', label=f'image {nimg}', color=color)
    ax.plot(x_pdf, theoretical_pdf, label=f"{distr} distribution")

    fig = ax.get_figure()
    width, height = fig.get_size_inches()

    ax.set_xlabel('Area (μm²)', fontsize=width)
    ax.set_ylabel('Density', fontsize=width)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=width * 0.75)
    ax.legend(fontsize=width * 0.75)



