'''
Image binarization
'''

import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_triangle
from scipy.ndimage import binary_fill_holes
import cv2

nimg = 3
show = True

try:
  image_name = os.path.join('Images', f'Image {nimg} (clean)')
  image = Image.open(f'{image_name}.jpg').convert("L")
except:
  image_name = os.path.join('Images', f'Image {nimg}')
  image = Image.open(f'{image_name}.jpg').convert("L")

img = np.asarray(image, dtype='uint8')

threshold = threshold_triangle(img)
binary = (img > threshold)*1

filled = binary_fill_holes(binary).astype('uint8')

kernel = np.ones((2, 2),np.uint8)
opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

if show:
  plt.imshow(img, cmap='gray')
  plt.show()
  plt.imshow(binary, cmap='gray')
  plt.show()
  plt.imshow(filled, cmap='gray')
  plt.show()
  plt.imshow(opened, cmap='gray')
  plt.show()

save_dir = 'Binarized images'
if save_dir:
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f"Binarized image {nimg}.jpg")

  to_export = (opened>0)*255
  to_export = to_export.astype('uint8')

  img_to_save = Image.fromarray(to_export, mode='L')
  img_to_save.save(save_path)