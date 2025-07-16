"""
Utility functions for image analysis pipeline.
"""

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
from skimage.segmentation import clear_border
from scipy.ndimage import label as scipylabel
from scipy.stats import ks_1samp, lognorm, pareto
import json
import os

def load_image_settings(path):

    with open(path, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def show_or_save_fig(fig, show=False, save_dir=None, figname=None):

    fig.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, f"{figname}.png")
        fig.savefig(save_path)
    if show:
        plt.show()

    plt.close(fig)


def import_binarized_img(nimg, show=False, save_dir=None):

    image_path = os.path.join("Binarized images", f"Binarized image {nimg}.jpg")
    try:
        image = Image.open(image_path).convert("L")
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

    image_array = np.asarray(image, dtype='uint8')
    binarized = (image_array > 128).astype('uint8')

    image_settings = load_image_settings('image_settings.json')
    settings = image_settings.get(nimg)

    if show or save_dir:
        figname = f'image_{nimg}_binarized'
        fig, ax = plt.subplots()
        ax.imshow(binarized, cmap='gray')
        ax.set_title('Binarized image')
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    return binarized, settings


def labeling_objects(nimg, binarized, cleared=False, show=False, save_dir=None):

    if cleared:
        border_cleared = clear_border(binarized)
    else:
        border_cleared = binarized

    labeled_image, num_labels = scipylabel(border_cleared)

    if show or save_dir:
        figname = f'image_{nimg}_labeled'
        fig, ax = plt.subplots()
        ax.imshow(labeled_image)
        ax.set_title('Labeled image')
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    return labeled_image, num_labels, border_cleared


def store_objects_info(nimg, labeled_image, num_labels, min_area, show=False, save_dir=None):

    objects_info = {'label': [], 'area': [], 'perimeter': []}
    filtered_image = np.copy(labeled_image)

    for lab in range(1, num_labels + 1):
        object_mask = (labeled_image == lab).astype('uint8')
        area = np.sum(object_mask)

        contours, _ = cv2.findContours(object_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True)

        if (area <= min_area) or (perimeter <= 1):
            filtered_image[labeled_image == lab] = 0
            continue

        objects_info['label'].append(lab)
        objects_info['area'].append(area)
        objects_info['perimeter'].append(perimeter)

    filtered_image = (filtered_image > 0).astype('uint8')

    if show or save_dir:
        figname = f'image_{nimg}_filtered'
        fig, ax = plt.subplots()
        ax.imshow(filtered_image, cmap='gray')
        ax.set_title('Filtered image')
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    return objects_info, filtered_image

def import_label_and_store(nimg, cost_min_area, show=False, save_dir=None):

    binarized, settings = import_binarized_img(nimg, show=show, save_dir=save_dir)
    labeled_image, num_labels, border_cleared = labeling_objects(nimg, binarized, cleared=False, show=show, save_dir=save_dir)
    if cost_min_area:
        min_area = 1
    else:
        min_area = settings["min_area"] * ((settings["conv_factor"])**2) or 1
    objects, filtered_image = store_objects_info(nimg, labeled_image, num_labels, min_area, show=show, save_dir=save_dir)
    return objects, filtered_image, settings

def fractal_dimension_measure(nimg, areas, perimeters, show=False, save_dir=None):

    log_areas = np.log(areas)
    log_perimeters = np.log(perimeters)
    fit_params, cov = np.polyfit(log_areas, log_perimeters, 1, cov=True)
    m, q = fit_params
    m_err = np.sqrt(cov[0, 0])
    fract_dim = m * 2
    fract_dim_err = m_err * 2

    if show or save_dir:
        figname = f'image_{nimg}_fractaldim_plot'
        fig, ax = plt.subplots()
        ax.plot(log_areas, log_perimeters, '.')
        x = np.linspace(min(log_areas), max(log_areas), 10)
        y = m * x + q
        ax.plot(x, y, '-')
        ax.set_xlabel('LogA')
        ax.set_ylabel('LogP')
        ax.set_title('Fractal dimension plot')
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    return fract_dim, fract_dim_err


def compute_surface_coverage(img):
    area_pixel = np.sum(img)
    total_pixel = img.size
    surface_coverage = (area_pixel / total_pixel) * 100
    return surface_coverage


def round_value(value):
    if value == 0:
        return 0

    if value == None:
        return None

    order = int(np.floor(np.log10(abs(value))))
    decimals = -(order-1)

    if decimals < 0:
        decimals = 0

    rounded_value = round(value, decimals)
    return rounded_value


def round_value_by_error(value, error):
    if error == 0:
        return round(value, 3), 0

    order = int(np.floor(np.log10(abs(error))))
    decimals = -order

    if decimals < 0:
        decimals = 0

    rounded_error = round(error, decimals)
    rounded_value = round(value, decimals)
    return rounded_value, rounded_error


def find_threshold_ks(nimg, a, maximum, conv_factor, distr, show=False, save_dir=None):

    ks_pvalues = []
    params_list = []

    start, stop, step = 0, maximum/(conv_factor)**2, 5/(conv_factor)**2
    t_head = np.arange(start, stop, step)

    for i in t_head:
        d = a[(a>=i)]

        if distr == 'lognorm':
            params_ln = lognorm.fit(d)
            stat_ln, pval_ln = ks_1samp(d, lambda x: lognorm.cdf(x, s=params_ln[0], loc=params_ln[1], scale=params_ln[2]))
            params, pval = params_ln, pval_ln
        elif distr == 'pareto':
            params_p = pareto.fit(d)
            stat_p, pval_p = ks_1samp(d, lambda x: pareto.cdf(x, b=params_p[0], loc=params_p[1], scale=params_p[2]))
            params, pval = params_p, pval_p
        else:
            print('define the distribution inside the function')
            break

        ks_pvalues.append(pval)
        params_list.append([float(np.round(params[0], 3)), float(np.round(params[1], 3)), float(np.round(params[2], 3))])

    if show or save_dir:
        figname = f'image_{nimg}_KS_results_plot'
        fig, ax = plt.subplots()
        ax.plot(t_head, ks_pvalues, '-b',marker='o', label=distr)
        ax.axhline(y=0.05, color='r', linestyle='--')
        ax.set_xlabel(r'threshold ($μm^2$)')
        ax.set_ylabel('p-value')
        ax.set_title('KS results')
        ax.grid()
        show_or_save_fig(fig=fig, show=show, save_dir=save_dir, figname=figname)

    try:
        point = np.where(np.array(ks_pvalues) >= 0.05)[0][0]
        thra = t_head[point]
        thra = round_value(thra)
        par = params_list[point]
    except:
        print('no value above p-value')
        thra = None
        par = [None, None, None]

    return thra, par


