import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import data, exposure, img_as_float
import scipy.misc
import os
from skimage.exposure import rescale_intensity
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage.filters import rank
from skimage.morphology import disk


def sobel_each(image):
    return filters.sobel(image)

def sobel_hsv(image):
    return filters.sobel(image)

rootdir = 'coins/'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))
        img_path = os.path.join(subdir, file)
        im = imread( img_path)
        img_gray = rgb2gray(im)
        img_gray.min()

        v_min, v_max = np.percentile(img_gray, (0.2, 99.8))
        better_contrast = exposure.rescale_intensity(img_gray, in_range=(v_min, v_max))

        selem = disk(30)
        img_eq = rank.equalize(better_contrast, selem=selem)
        img_rescale = exposure.equalize_hist(better_contrast)

        save_directory = 'CNN_data/histImages/'+subdir+"/"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        scipy.misc.imsave(save_directory+file,  better_contrast)









