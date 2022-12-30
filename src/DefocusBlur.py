# -*- coding: utf-8 -*-
#taken from: https://github.com/lospooky/pyblur
#fixed by https://github.com/jayroxis/blurlab
#MIT LICENSE - https://github.com/lospooky/pyblur/blob/master/LICENSE
#Copyright (c) 2016 Simone Cirillo

import numpy as np
from PIL import Image
from skimage.draw import disk
import cv2

defocusKernelDims = [3,5,7,9]

def DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))    
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)

def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    # convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    convolved = cv2.filter2D(imgarray, -1, kernel).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = dim / 2
    circleRadius = circleCenterCoord +1
    
    rr, cc = disk((circleCenterCoord, circleCenterCoord), circleRadius-1)
    kernel[rr, cc] = 1
    
    if(dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)
        
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel

def Adjust(kernel, kernelwidth):
    kernel[0,0] = 0
    kernel[0,kernelwidth-1]=0
    kernel[kernelwidth-1,0]=0
    kernel[kernelwidth-1, kernelwidth-1] =0 
    return kernel