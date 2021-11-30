import numpy as np                 # This contains all our math functions we'll need
# This toolbox is what we'll use for reading and writing images
import skimage.io as io
# %matplotlib notebook
# This toolbox is to create our plots. Line above makes them interactive
import matplotlib.pyplot as plt
import matplotlib.patches as patches #for plotting rectanglular bounding box over features in images
import seaborn as sns              #plotting tool
import matplotlib.ticker as ticker
import colorcet as cc       #set of color maps that can be called as strings in plt
import cmasher as cmr       #set of color maps that can be called as strings in plt
from colormath.color_objects import *
from colormath.color_conversions import convert_color
from matplotlib import cm       #colormaps
from matplotlib.colors import ListedColormap      #for creating colormaps from a list of RGB values
import os       # This toolbox is a useful directory tool to see what files we have in our folder
import cv2                         # image processing toolbox
import glob as glob                # grabbing file names
import czifile                     # read in the czifile
from skimage import morphology, util, filters
from scipy import stats, optimize, ndimage               # for curve fitting 
from scipy.signal import medfilt, convolve   # Used to detect overlap
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label, regionprops      # For labeling regions in thresholded images and calculating properties of labeled regions
from skimage.segmentation import clear_border       # Removes junk from the borders of thresholded images
from skimage.color import label2rgb  # Pretty display labeled images
from skimage.morphology import opening, disk, dilation, remove_small_objects, remove_small_holes     # morphology operations
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, generate_binary_structure                    # morphological operations
from skimage.feature import peak_local_max      #finds local peaks based on a local or absolute threshold
import pandas as pd  # For creating our dataframe which we will use for filtering
from pandas import DataFrame, Series #for convenience
import untangle   # for parsing the XML files
from image_plotting_tools import *
from candle_functions import get_czifile_metadata



def prep_file_zproject_actomyosin(czifilename,show_images=True,save_images=True):
    # read in the file
    imstack = czifile.imread(czifilename)

#     # keep only those dimensions that have the z-stack
    imstack_act = imstack[0, 0, 0, :, :, :, :, 0]
    imstack = imstack[0, 0, 1, :, :, :, :, 0]
    
    #third dimension is channel
    #fourth dimension is time
    #fifth dimension is z
    #sixth and seventh are x and y

    # make max projection 
    imstack_max = np.max(imstack, axis=1)
    #make sum projection
    imstack_sum = np.sum(imstack, axis=1)
    # make max projection 
    imstack_act_max = np.max(imstack_act, axis=1)
    #make sum projection
    imstack_act_sum = np.sum(imstack_act, axis=1)
    #make a actin-bleedthrough subtraction myosin sum image
    imstack_sum_sub = imstack_sum - (0.1*imstack_act_sum)
    imstack_sum_sub [imstack_sum_sub <0] = 0

    # get the metadata
    exp_details = get_czifile_metadata(czifilename)
    
    if show_images:
        imstack_max_fig, imstack_max_axes = plt.subplots()
        imstack_max_axes.imshow(imstack_max[0], cmap=qbk_cmap, vmin=50, vmax=np.max(imstack_max)*.4)
        imstack_max_fig.show()
        imstack_act_max_fig, imstack_act_max_axes = plt.subplots()
        imstack_act_max_axes.imshow(imstack_act_max[0], cmap=qbk_cmap, vmin=50, vmax=np.max(imstack_act_max)*.4)
        imstack_act_max_fig.show()
        imstack_sum_sub_fig, imstack_sum_sub_axes = plt.subplots()
        imstack_sum_sub_axes.imshow(imstack_sum_sub[0], cmap=qbk_cmap, vmin=50, vmax=np.max(imstack_sum_sub)*.4)
        imstack_sum_sub_fig.show()
        
    if save_images:
        #save the z projections if they aren't already saved
        io.imsave(czifilename[:-4] + '_max.tif', imstack_max)
        io.imsave(czifilename[:-4] + '_sum.tif', imstack_sum)
        io.imsave(czifilename[:-4] + '_act_max.tif', imstack_act_max)
        io.imsave(czifilename[:-4] + '_act_sum.tif', imstack_act_sum)
        io.imsave(czifilename[:-4] + '_sum_sub.tif', imstack_sum_sub)
    
    return imstack, imstack_act, imstack_max, imstack_sum, imstack_act_max, imstack_act_sum, imstack_sum_sub, exp_details



def candle_masks(qbk_cmap, im_sum, threshold_factor= 0.5, show_results=False):
    
    intensity_values = np.unique(im_sum.ravel())
    # reduce list of intensity values down to something manageable to speed up computation
    if len(intensity_values) > 300:
        slice_width = np.round(len(intensity_values)/300).astype('int')
        if slice_width == 0:
            slice_width = 1
        intensity_values = intensity_values[::slice_width]
    # Find the mean intensity value of the image
    intensity_mean = np.mean(im_sum)
    intensity_difference = []
    # create a zero matrix to hold our difference values
    for i,intensity in enumerate(intensity_values):
        # make a mask of pixels about a given intensity
        mask = im_sum > intensity
        intensity_difference.append(np.sum(im_sum[mask]) - intensity_mean*np.sum(mask))
    # find the maximum value of the intensity_difference and set it equal to the threshold
    max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
    threshold = intensity_values[max_intensity[0][0]]
#     print(threshold)
    # make a mask at this threshold
    mask = im_sum > threshold * threshold_factor
    small_object_size = 11 * 11
    # get rid of small objects
    mask = remove_small_objects(mask, small_object_size)
    mask = remove_small_holes(mask, 10000)
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    areas = []
    for region in props:
        areas.append(region.area)
    max_area_label = np.argwhere(areas == np.max(areas))
    cytoplasm_mask = labeled_mask == max_area_label[0][0] + 1
    SE = disk(5)
#     cytoplasm_mask = binary_dilation(cytoplasm_mask, structure=SE)
    cell_mask = binary_fill_holes(cytoplasm_mask)
    extracellular_mask = cell_mask == False
    
    if show_results:
        mask_fig, mask_ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
        mask_ax[0,0].imshow(cell_mask)
        mask_ax[0,0].set_title('Cell Mask')
        mask_ax[0,1].imshow(cytoplasm_mask)
        mask_ax[0,1].set_title('Cytoplasm Mask')
        mask_ax[1,0].imshow(extracellular_mask)
        mask_ax[1,0].set_title('Extracellular Mask')
        mask_ax[1,1].imshow(im_sum, cmap=qbk_cmap)
        mask_ax[1,1].set_title('Actin Image')
        for ax in mask_ax.ravel():
            ax.axis('off')
        mask_fig.show()
        
    counts, bins = np.histogram(im_sum[cell_mask], bins=150)
    bins = bins[:-1] + np.diff(bins/2)
    hist_max = np.argwhere(counts == np.max(counts))
    bg_pixel = bins[hist_max[0, 0]]
    frame_std = np.std(im_sum[cell_mask])


    
    return cell_mask, cytoplasm_mask, extracellular_mask, bg_pixel, frame_std



def myo_masks(qbk_cmap, im_sum, threshold_factor= 1.8, show_results=True):
    intensity_values = np.unique(im_sum.ravel())
    # reduce list of intensity values down to something manageable to speed up computation
    if len(intensity_values) > 300:
        slice_width = np.round(len(intensity_values)/300).astype('int')
        if slice_width == 0:
            slice_width = 1
        intensity_values = intensity_values[::slice_width]
    # Find the mean intensity value of the image
    intensity_mean = np.mean(im_sum)
    intensity_difference = []
    # create a zero matrix to hold our difference values
    for i,intensity in enumerate(intensity_values):
        # make a mask of pixels about a given intensity
        mask = im_sum > intensity
        intensity_difference.append(np.sum(im_sum[mask]) - intensity_mean*np.sum(mask))
    # find the maximum value of the intensity_difference and set it equal to the threshold
    max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
    threshold = intensity_values[max_intensity[0][0]]
    # make a mask at this threshold
    mask = im_sum > threshold * threshold_factor
    small_object_size = 21 * 21
    # get rid of small objects
    mask = remove_small_objects(mask, small_object_size)
    mask = remove_small_holes(mask, 1500)
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    areas = []
    for region in props:
        areas.append(region.area)
    max_area_label = np.argwhere(areas == np.max(areas))
    cytoplasm_mask = labeled_mask == max_area_label[0][0] + 1
    SE = disk(3)
#     cytoplasm_mask = binary_dilation(cytoplasm_mask, structure=SE)
    cytoplasm_mask = binary_dilation(cytoplasm_mask, structure=SE)
#     myo_mask = binary_fill_holes(cytoplasm_mask)
    myo_mask = cytoplasm_mask
    myo_mask_invert =  myo_mask == False
    labeled_mask = label(myo_mask_invert)
    props = regionprops(labeled_mask)
    areas = []
    for region in props:
        areas.append(region.area)
    max_area_label = np.argwhere(areas == np.max(areas))
    myo_mask_invert = labeled_mask == max_area_label[0][0] + 1
    nuclear_mask = im_sum < threshold * .8
    small_object_size = 21 * 21
    # get rid of small objects
    nuclear_mask = remove_small_objects(nuclear_mask, small_object_size)
    nuclear_mask = remove_small_holes(nuclear_mask, 150000)
    SE = disk(3)
    nuclear_mask = binary_dilation(nuclear_mask, structure=SE)
    nuclear_mask = binary_fill_holes(nuclear_mask)
    small_object_size = 51 * 51
    # get rid of small objects
#     nuclear_mask = remove_small_objects(nuclear_mask, small_object_size)
    labeled_mask = label(nuclear_mask)
    props = regionprops(labeled_mask)
    areas = []
    for region in props:
        areas.append(region.area)
    max_area_label = np.argwhere(areas == np.max(areas))
    nuclear_mask = labeled_mask == max_area_label[0][0] 

    if show_results:
        mask_fig, mask_ax = plt.subplots(nrows=2, ncols=2,figsize=(10,10))
        mask_ax[0,0].imshow(myo_mask)
        mask_ax[0,0].set_title('Myosin Mask')
        mask_ax[0,1].imshow(myo_mask_invert)
        mask_ax[0,1].set_title('Myosin Mask invert')
        mask_ax[1,0].imshow(nuclear_mask)
        mask_ax[1,0].set_title('Nuclear Mask')
        mask_ax[1,1].imshow(im_sum, cmap=qbk_cmap, vmin=50, vmax=10000)
        mask_ax[1,1].set_title('Image')
        for ax in mask_ax.ravel():
            ax.axis('off')
        mask_fig.show()
    return myo_mask, myo_mask_invert



def make_protrusion_masks(qbk_cmap, czifilename, imstack_sum, imstack_act_max, imstack_sum_sub, act_threshold_factor=0.5, myo_threshold_factor=1.8, test_frame=0, test_params=True, save_results=False, show_results=True):
    protrusion_mask = np.zeros_like(imstack_sum)
    protrusion_movie = np.zeros_like(imstack_sum)
    if test_params:
        f = test_frame
        im_act = imstack_act_max[f]
        im_myo = imstack_sum_sub[f] 
        cell_mask, cytoplasm_mask, extracellular_mask, bg_pixel, frame_std = candle_masks(qbk_cmap, im_act, threshold_factor= act_threshold_factor, show_results=False)
        myo_mask, myo_mask_invert = myo_masks(qbk_cmap, im_myo, threshold_factor= myo_threshold_factor, show_results=False)
        protrusion = (myo_mask_invert) * cell_mask
        protrusion_mask[f] = protrusion
        protrusion_movie[f] = imstack_sum_sub[f] * protrusion
    else:
        for f in range(len(imstack_sum)):
            im_act = imstack_act_max[f]
            im_myo = imstack_sum_sub[f] 
            cell_mask, cytoplasm_mask, extracellular_mask, bg_pixel, frame_std = candle_masks(qbk_cmap, im_act, threshold_factor= act_threshold_factor, show_results=False)
            myo_mask, myo_mask_invert = myo_masks(qbk_cmap, im_myo, threshold_factor= myo_threshold_factor, show_results=False)
            protrusion = (myo_mask_invert) * cell_mask
            protrusion_mask[f] = protrusion
            protrusion_movie[f] = imstack_sum_sub[f] * protrusion
        
    if save_results:
        io.imsave(czifilename[:-4] + '_sub_protrusion_mask.tif', protrusion_mask)
        io.imsave(czifilename[:-4] + '_sub_protrusion_movie.tif', protrusion_movie)
        
    if show_results:
        mask_fig, mask_ax = plt.subplots(ncols=3,figsize=(15,5))
        mask_ax[0].imshow(protrusion_mask[test_frame])
        mask_ax[0].set_title('Protrusion Mask')
        mask_ax[1].imshow(protrusion_movie[test_frame], cmap=qbk_cmap)
        mask_ax[1].set_title('Protrusion Movie')
        mask_ax[2].imshow(imstack_sum_sub[test_frame], cmap=qbk_cmap, vmin=50, vmax=10000)
        mask_ax[2].set_title('Actin Subtracted Myosin Movie')

    
    return protrusion_mask, protrusion_movie


'''average the pixel intensities over a given number of frames and then keep only the pixels that are in the majority of those frames to smooth the mask. 
Window size is the number of frames on either side of the frame of interest (i.e. window=3 is a 7 frame moving average)'''
def protrusion_average(czifilename, protrusion_mask, imstack_sum_sub, imstack_sum, protrusion_movie, window=3, save_results=True):
    n_frames = len(imstack_sum)
    protrusion_mask_avg = np.zeros_like(imstack_sum)
    protrusion_movie = np.zeros_like(imstack_sum)
    for n in range(window,n_frames-(window-1)):
        frame_avg = np.sum(protrusion_mask[n-window:n+window], axis=0)
        frame_avg = frame_avg > window+2
        protrusion_mask_avg[n] =frame_avg
        protrusion_movie[n] = imstack_sum_sub[n] * frame_avg
    protrusion_mask_avg = protrusion_mask_avg[window:(1-window)]
    protrusion_movie = protrusion_movie[window:(1-window)]
    moving_avg = (window*2)+1
    if save_results:
        io.imsave(czifilename[:-4] + '_sub_protrusion_mask_'+str(moving_avg)+'avg.tif', protrusion_mask_avg)
        io.imsave(czifilename[:-4] + '_sub_protrusion_movie_'+str(moving_avg)+'avg.tif', protrusion_movie)
    return protrusion_mask_avg, protrusion_movie



def make_cluster_mask(qbk_cmap, czifilename, protrusion_mask,protrusion_mask_avg, protrusion_movie, threshold_factor=30, test_frame=100, test_params=True, show_results=True, save_results=True):
    hist_bg_list = []
    avg_bg_list = []
    bg_std_list = []
    cluster_mask = np.zeros_like(protrusion_movie)
    cluster_mask_invert = np.zeros_like(protrusion_movie)
    protusion_cytoplasm = np.zeros_like(protrusion_movie)
    protusion_cytoplasm_movie = np.zeros_like(protrusion_movie)
    if test_params:
        f=test_frame
        im_sum = protrusion_movie[f].copy()
        intensity_values = np.unique(im_sum.ravel())

        # reduce list of intensity values down to something manageable to speed up computation
        if len(intensity_values) > 300:
            slice_width = np.round(len(intensity_values)/300).astype('int')
            if slice_width == 0:
                slice_width = 1
            intensity_values = intensity_values[::slice_width]
        # Find the mean intensity value of the image
        intensity_mean = np.mean(im_sum)
        intensity_difference = []
        # create a zero matrix to hold our difference values
        for i,intensity in enumerate(intensity_values):
            # make a mask of pixels about a given intensity
            mask = im_sum > intensity
            intensity_difference.append(np.sum(im_sum[mask]) - intensity_mean*np.sum(mask))
        # find the maximum value of the intensity_difference and set it equal to the threshold
        max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
        threshold = intensity_values[max_intensity[0][0]]
        # make a mask at this threshold
        mask = im_sum > threshold * threshold_factor
        small_object_size = 5 * 5
        # get rid of small objects
        mask = remove_small_objects(mask, small_object_size)
        SE = disk(1)
        cluster_mask_frame = binary_dilation(mask, structure=SE)
        cluster_mask[f] = cluster_mask_frame
        cluster_mask_invert[f] = cluster_mask_frame ==False
        protusion_cytoplasm_frame = protrusion_mask_avg[f] * cluster_mask_invert[f]
        protusion_cytoplasm[f] = protusion_cytoplasm_frame
        protusion_cytoplasm_movie[f] = protrusion_movie[f] * protusion_cytoplasm_frame
        counts, bins = np.histogram(protusion_cytoplasm_movie[f], bins=150)
        bins = bins[:-1] + np.diff(bins/2)
        hist_max = np.argwhere(counts == np.max(counts))
        hist_bg = bins[hist_max[0, 0]]
        frame_std = np.std(protusion_cytoplasm_movie[f])
        avg_bg = np.mean(protusion_cytoplasm_movie[f])
        print(hist_bg)
        print(frame_std)
        print(avg_bg)


        
    else:
        for f in range(len(protrusion_movie)):
            im_sum = protrusion_movie[f].copy()
            intensity_values = np.unique(im_sum.ravel())

            # reduce list of intensity values down to something manageable to speed up computation
            if len(intensity_values) > 300:
                slice_width = np.round(len(intensity_values)/300).astype('int')
                if slice_width == 0:
                    slice_width = 1
                intensity_values = intensity_values[::slice_width]
            # Find the mean intensity value of the image
            intensity_mean = np.mean(im_sum)
            intensity_difference = []
            # create a zero matrix to hold our difference values
            for i,intensity in enumerate(intensity_values):
                # make a mask of pixels about a given intensity
                mask = im_sum > intensity
                intensity_difference.append(np.sum(im_sum[mask]) - intensity_mean*np.sum(mask))
            # find the maximum value of the intensity_difference and set it equal to the threshold
            max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
            threshold = intensity_values[max_intensity[0][0]]
            # make a mask at this threshold
            mask = im_sum > threshold * threshold_factor
            small_object_size = 5 * 5
            # get rid of small objects
            cluster_mask_frame = remove_small_objects(mask, small_object_size)
#             SE = disk(1)
#             cluster_mask_frame = binary_dilation(cluster_mask_frame, structure=SE)
            cluster_mask[f] = cluster_mask_frame
            cluster_mask_invert[f] = cluster_mask_frame ==False
            protusion_cytoplasm_frame = protrusion_mask_avg[f] * cluster_mask_invert[f]
            protusion_cytoplasm[f] = protusion_cytoplasm_frame
            protusion_cytoplasm_movie[f] = protrusion_movie[f] * protusion_cytoplasm_frame
            counts, bins = np.histogram(protusion_cytoplasm_movie[f], bins=150)
            bins = bins[:-1] + np.diff(bins/2)
            hist_max = np.argwhere(counts == np.max(counts))
            hist_bg = bins[hist_max[0, 0]]
            frame_std = np.std(protusion_cytoplasm_movie[f])
            avg_bg = np.mean(protusion_cytoplasm_movie[f])
            hist_bg_list.append(hist_bg)
            avg_bg_list.append(frame_std)
            bg_std_list.append(avg_bg)


    
    if show_results:
        mask_fig, mask_ax = plt.subplots(ncols=3, figsize=(25,10))
        mask_ax[0].imshow(cluster_mask[test_frame])
        mask_ax[0].set_title('Cluster Mask')
        mask_ax[1].imshow(protusion_cytoplasm[test_frame])
        mask_ax[1].set_title('Protrusion Cytoplasm')
        mask_ax[2].imshow(protrusion_movie[test_frame],cmap=qbk_cmap, vmin=1000, vmax=10000)
        mask_ax[2].set_title('Image')
    if save_results:
        io.imsave(czifilename[:-4] + '_cluster_mask.tif', cluster_mask)
        io.imsave(czifilename[:-4] + '_protusion_cytoplasm_movie.tif', protusion_cytoplasm_movie)
        
        
    return cluster_mask, hist_bg_list, avg_bg_list, bg_std_list, protusion_cytoplasm, protusion_cytoplasm_movie




def cluster_mask_avg(czifilename, cluster_mask, window=3, save_results=True):
    n_frames = len(cluster_mask)
    cluster_mask_avg = np.zeros_like(cluster_mask)

    for n in range(window,n_frames-(window-1)):
        frame_avg = np.sum(cluster_mask[n-window:n+window], axis=0)
        frame_avg = frame_avg > window+2
        cluster_mask_avg[n] =frame_avg
    cluster_mask_avg = cluster_mask_avg[window:(1-window)]
    moving_avg = (window*2)+1

    if save_results:
        io.imsave(czifilename[:-4] + '_cluster_mask_'+str(moving_avg)+'avg.tif', cluster_mask_avg)

    return cluster_mask_avg



