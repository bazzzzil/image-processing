# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:37:29 2020

@author: Layale

Image Processing Utility Functions

For example implementation, see example.ipynb
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d as conv2

def load_data(filename):
    """
    Function to load in image data. Can be .mat, .jpg, .png, .tiff.
    Note: only one channel is passed along after reading in.

    Parameters
    ----------
    filename : string
        Name of file in working directory

    Returns
    -------
    loaded_data : 2D array
        Loaded image data
    """

    ext = os.path.splitext(filename)[1]

    if ext == '.mat':
        # loadmat loads the .mat file as a dictionary with the keys
        # dict_keys(['__header__', '__version__', '__globals__', key_var])
        # We want the last key (key_var) which holds the variable that 
        # stores the image

        file_contents = sio.loadmat(filename)
        key_var = list(file_contents.keys())[-1]
        return np.squeeze(file_contents[key_var])

    else:
        # Pass as gray image
        image = cv.imread(filename, 0)
        return np.squeeze(image)


def plot_img(data, plot_title="Image"):
    """
    Plotting function to reduce clutter.

    Parameters
    ----------
    data : 2D array
        Image to be shown
        
    *plot_title : string
        Title for plot

    Returns
    -------
    Nothing.
    """
    plt.imshow(data)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title(plot_title)

def ifft2d(kdata):
    """
    Does fftshift(ifft2(ifftshift(kdata))) to reduce clutter. Only for 2D arrays.

    Parameters
    ----------
    kdata : 2D complex array
        kspace data

    Returns
    -------
    idata : 2D complex array
        image space data
    """    
    return fftshift(ifft2(ifftshift(kdata)))

def fft2d(image):
    """
    Does ifftshift(fft2(fftshift(image))) to reduce clutter. Only for 2D arrays.

    Parameters
    ----------
    image : 2D complex/real array
        image space data

    Returns
    -------
    kdata : 2D complex array
        kspace data
    """    
    return ifftshift(fft2(fftshift(image)))

def zero_padder(data, zero_dim):
    """
    Function to zeropad 2D or 3D data for use in 
    various imaging processing techniques.
    Data can be complex or real.

    Parameters
    ----------
    data : 2D or 3D complex/real array

    zero_dim : 1D array containing desired final dimensions

    Returns
    -------
    padded_data : zero-padded 2D or 3D complex/real array

    Examples
    -------
    % example 2D: padded_data = zero_padder(data, 128, 128);
    % example 3D: padded_data = zero_padder(data, 128, 128, 128);
    -------

    """    
    data_shape = np.shape(data)

    if len(zero_dim)==2:

        dim1 = zero_dim[0]
        dim2 = zero_dim[1]

        if np.iscomplexobj(data):
            padded_data = np.zeros([dim1, dim2], dtype=complex)

        else:
            padded_data = np.zeros([dim1, dim2])

        leftDim1 = round(dim1 / 2) - round(data_shape[0] / 2) 
        rightDim1 = round(dim1 / 2) + round(data_shape[0] / 2)
        leftDim2 = round(dim2 / 2) - round(data_shape[1] / 2) 
        rightDim2 = round(dim2 / 2) + round(data_shape[1] / 2)

        if all(x % 2 == 0 for x in list(data_shape)):
            padded_data[leftDim1:rightDim1, leftDim2:rightDim2] = data
        else:
            try:
                padded_data[leftDim1:rightDim1+1, leftDim2:rightDim2+1] = data
            except:
                 padded_data[leftDim1+1:rightDim1, leftDim2+1:rightDim2] = data

    elif len(zero_dim)==3:

        dim1 = zero_dim[0]
        dim2 = zero_dim[1]
        dim3 = zero_dim[2]

        if np.iscomplexobj(data):
            padded_data = np.zeros([dim1, dim2, dim3], dtype=complex)

        else:
            padded_data = np.zeros([dim1, dim2, dim3])
        
        leftDim1 = round(dim1 / 2) - round(data_shape[0] / 2)
        rightDim1 = round(dim1 / 2) + round(data_shape[0] / 2)
        leftDim2 = round(dim2 / 2) - round(data_shape[1] / 2)
        rightDim2 = round(dim2 / 2) + round(data_shape[1] / 2)
        leftDim3 = round(dim3 / 2) - round(data_shape[2] / 2)
        rightDim3 = round(dim3 / 2) + round(data_shape[2] / 2)
        
        if all(x % 2 == 0 for x in list(data_shape)):
            padded_data[leftDim1-1:rightDim1, leftDim2-1:rightDim2, leftDim3-1:rightDim3] = data
        else:
            try:
                padded_data[leftDim1:rightDim1+1, leftDim2:rightDim2+1, leftDim3:rightDim3+1] = data
            except:
                 padded_data[leftDim1+1:rightDim1, leftDim2+1:rightDim2, leftDim3+1:rightDim3] = data

    return padded_data

def low_pass_avg(data):
    """
    Performs a low pass filtering using an averaging filter.
    Filtering is performed via convolution with a kernel.
    
    Parameters
    ----------
    data : 2D complex/real array

    Returns
    -------
    image with low pass filter applied
    """    
    if np.iscomplexobj(data):
        low_pass = np.array([[1/8, 1/8, 1/8],
                                    [1/8,  8,  1/8],
                                    [1/8, 1/8, 1/8]], dtype=complex)
 
    else:
        low_pass = np.array([[1/8, 1/8, 1/8],
                                    [1/8,  8,  1/8],
                                    [1/8, 1/8, 1/8]])

    return conv2(data, low_pass,'same')

def low_pass_gauss(data, filter_width=10):
    """
    Performs a low pass filtering using a Gaussian filter.
    Filtering is performed via convolution with a kernel.

    Can control width of filter to determine degree of blurring.

    Parameters
    ----------
    data : 2D complex/real array

    Returns
    -------
    image with low pass filter applied
    """    

    # Creating a custom 5x5 low-pass Gaussian filter
    if np.iscomplexobj(data):
        [y,x] = np.meshgrid(np.arange(-2,3,1), np.arange(-2,3,1),dtype=complex)
 
    else:
        [y,x] = np.meshgrid(np.arange(-2,3,1), np.arange(-2,3,1))

    low_pass=np.exp(-(x**2+y**2)/(2*filter_width**2))

    return conv2(data, low_pass,'same')

def x_grad(data):
    """
    Applies x gradient filter across image.
    Filtering is performed via convolution with a kernel.

    Parameters
    ----------
    data : 2D complex/real array

    Returns
    -------
    image with x gradient filter applied
    """    
    # Creating custom 3x3 Y and X gradient filters
    if np.iscomplexobj(data):
        [_,xgrad] = np.meshgrid(np.arange(-1,2,1), np.arange(-1,2,1),dtype=complex)
 
    else:
        [_,xgrad] = np.meshgrid(np.arange(-1,2,1), np.arange(-1,2,1))

    return conv2(data, xgrad,'same')

def y_grad(data):
    """
    Applies y gradient filter across image.
    Filtering is performed via convolution with a kernel.

    Parameters
    ----------
    data : 2D complex/real array

    Returns
    -------
    image with x gradient filter applied
    """    

    # Creating custom 3x3 Y and X gradient filters
    if np.iscomplexobj(data):
        [ygrad,_] = np.meshgrid(np.arange(-1,2,1), np.arange(-1,2,1),dtype=complex)
 
    else:
        [ygrad,_] = np.meshgrid(np.arange(-1,2,1), np.arange(-1,2,1))

    return conv2(data, ygrad,'same')

def high_pass(data):
    """
    Performs a high pass filtering for edge enhancement.
    Filtering is performed via convolution with a kernel.

    Parameters
    ----------
    data : 2D complex/real array

    Returns
    -------
    image with high pass filter applied
    """    
    # Creating custom 5x5 high pass filter
    # Trick is to make sure elements sum to zero
    if np.iscomplexobj(data):
        high_pass = np.ones([5,5],dtype=complex)
        high_pass[2,2] = -24
 
    else:
        high_pass = np.ones([5,5])
        high_pass[2,2] = -24

    return conv2(data, high_pass,'same')

def img_mask(data, threshold):
    """
    Returns a mask based on a threshold of the 
    original image.

    Parameters
    ----------
    data : 2D complex/real array
    threshold: float value between 0 and 1

    Returns
    -------
    mask : 2D logical array
    """    

    return data < max(data.flatten())*threshold

def svd_compress(data, num_singval=100):
    """
    Returns compressed image based off SVD
    algorithm. Can control level of compression.
    Only works with grayscale images.

    Parameters
    ----------
    data : 2D complex/real array
    num_singval: int value, number of singular values

    Returns
    -------
    new_image : 2D array (compressed image)
    """    
    (width, height) = np.shape(data)

    u, s, v = np.linalg.svd(data)

    W = np.diag(s)

    # We want to make sure the number of singular values kept is not 
    # greater than the image dimensions
    vecnum = min(num_singval, width, height)    

    # Here we reconstruct our image following the original equation for M = uSv
    new_image = u[:,:vecnum] @ W[:vecnum,:vecnum] @ v[:vecnum,:]

    return new_image
