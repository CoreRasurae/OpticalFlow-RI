#!/usr/bin/env python
"""
This file implements the Pyramidal topology for generic Optical Flow algorithms.
It supports a base optical flow algorithm and an optional optimizing optical flow algorithm.
The pyramidal level change is performed using a sub-pixel image shifting/warping method.
In addition, it is also possible to use iterations at the same pyramidal level.

//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2020-2021, Luís Mendes.
//
// @Authors
//    Luis Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
#from scipy.misc import imresize
import PIL
from PIL import Image
#from IPython.core import debugger
#debug = debugger.Pdb().set_trace

def imresize(im, res):
    return np.array(Image.fromarray(im).resize(res, PIL.Image.BILINEAR)) #PIL.Image.LANCZOS, PIL.Image.BICUBIC 

def updateNextPyramidalLevel(im1IterNext, im1IterPrev, Uaccum, Vaccum, U, V, warping=True, biLinear=True, scale=False):
    """
    The shift/warp image method was based on Liu & Shen published Physics Based Optical flow reference code.
    (https://github.com/Tianshu-Liu/OpenOpticalFlow)
    The idea is to shift the pixels in the first image by the same amount that was estimated by the previous
    pyramidal level including sub-pixel shifts, to align both images, pixel by pixel. This method enforces
    the initial velocity component for the new pyramidal level to be 0.0 in both components. Such procedure is
    also known as warping. In the Liu-Shen implementation they use the optical flow equation to estimate the
    pixel intensity variation due to the sub-pixel displacements. 
    An alternative method to estimate the fractional position pixel intensity is to use bi-linear interpolation,
    which seems to achieve better results than the optical flow equation for this specific purpose.
    This implementation supports both methods, as well as no warping. For no warping mode, this will simply 
    result in the BiCubic interpolation of the estimated velocities from the previous pyramidal level to the
    present level.

    Additional contract: this function must not resize or filter the images.
    :param im1IterNext: Input image for the next pyramidal level
    :param im1IterPrev: Image from the previous pyramidal level
    :param Uaccum: The accumulated component U velocity from the start of execution (1-st pyramidal Level)
    :param Vaccum: The accumulated component V velocity from the start of execution (1-st pyramidal Level)
    :param U: The component U velocity vector as estimated exclusively by the previous pyramidal level
    :param V: The component V velocity vector as estimated exclusively by the previous pyramidal level
    :param warping: True, Applies warping to the first image according to the computed displacement
    :param biLinear: True, use bi-linear interpolation to estimate the fractional position pixel intensity
                     False, use optical flow equation to estimate the fractional position pixel intensity
    :return the resulting image, the accumulated U displacement, the accumulated V displacement, the initial
            U and V displacements for the new pyramidal level
    """

    yDimLastStep = im1IterPrev.shape[0]
    xDimLastStep = im1IterPrev.shape[1]
    yDim = im1IterNext.shape[0]
    xDim = im1IterNext.shape[1]

    if xDimLastStep != xDim or yDimLastStep != yDim:
        usNew = imresize(Uaccum, (xDim, yDim))
        vsNew = imresize(Vaccum, (xDim, yDim))
    else:
        usNew=Uaccum
        vsNew=Vaccum

    if scale:
        scaleX = np.float32(xDim) / np.float32(xDimLastStep)
        scaleY = np.float32(yDim) / np.float32(yDimLastStep)
        usNew *= scaleX
        vsNew *= scaleY

    uInitialStep = None
    vInitialStep = None
    finalUAccum  = None
    finalVAccum  = None

    if warping:
        #Apply shifts/warping to image 1 - Move pixels at original position (i,j) to their new displaced destination
        ys=None
        xs=None
        limitX=im1IterNext.shape[1]-1
        limitY=im1IterNext.shape[0]-1
        if biLinear:
            ys=np.arange(0, im1IterNext.shape[0]-1)
            xs=np.arange(0, im1IterNext.shape[1]-1)
            xsMesh, ysMesh = np.meshgrid(xs,ys)
            usSwap = xsMesh + np.floor(usNew[0:-1,0:-1] + 0.5)
            vsSwap = ysMesh + np.floor(vsNew[0:-1,0:-1] + 0.5)
            limitX -= 1
            limitY -= 1
        else:
            ys=np.arange(0, im1IterNext.shape[0])
            xs=np.arange(0, im1IterNext.shape[1])
            xsMesh, ysMesh = np.meshgrid(xs,ys)
            usSwap = xsMesh + np.floor(usNew + 0.5)
            vsSwap = ysMesh + np.floor(vsNew + 0.5)
        usSwap = np.int16(usSwap);
        vsSwap = np.int16(vsSwap);
        usSwap[usSwap < 0] = 0;
        usSwap[usSwap >= limitX] = limitX
        vsSwap[vsSwap < 0] = 0;
        vsSwap[vsSwap >= limitY] = limitY       
        dUsNew = usNew - np.floor(usNew+0.5);
        dVsNew = vsNew - np.floor(vsNew+0.5);
 
        if biLinear:
            dUsNew = -dUsNew #Mirror fractional displacements to approximate Image1 to Image2
            dVsNew = -dVsNew #Mirror fractional displacements to approximate Image1 to Image2
            deltaY0=ysMesh
            deltaX0=xsMesh
            deltaY1=(ysMesh+1).astype(int)
            deltaX1=(xsMesh+1).astype(int)
            im1IterNext[vsSwap,usSwap] = (1 - dVsNew[0:-1,0:-1]) * (1 - dUsNew[0:-1,0:-1]) * im1IterNext[deltaY0,deltaX0] + \
                                         (1 - dVsNew[0:-1,0:-1]) *      dUsNew[0:-1,0:-1]  * im1IterNext[deltaY0,deltaX1] + \
                                              dVsNew[0:-1,0:-1]  * (1 - dUsNew[0:-1,0:-1]) * im1IterNext[deltaY1,deltaX0] + \
                                              dVsNew[0:-1,0:-1]  *      dUsNew[0:-1,0:-1]  * im1IterNext[deltaY1,deltaX1]
        else:
            #Apply integer displacement shifting/warping
            im1IterNext[vsSwap,usSwap] = im1IterNext[ysMesh,xsMesh];
            #Apply pixel intensity variations due to sub-pixel displacements (fractional part)
            #Soften spatial gradients of sub-pixel velocities, by gaussian filtering
            mask_size = 3
            dUsNew=gaussian_filter(dUsNew, 0.6*mask_size, truncate=4.0/0.6*mask_size)
            dVsNew=gaussian_filter(dVsNew, 0.6*mask_size, truncate=4.0/0.6*mask_size)
            #Estimate partial intensities variation due to sub-pixel displacements.
            dx=1
            dy=1
            dt=1
            #Use the optical flow equation to estimate the final pixel intensity due to the sub-pixel 
            #displacement in U and V components
            tempDx = (im1IterNext[0:-1,dx:]*dUsNew[0:-1,dx:]-im1IterNext[0:-1,0:-1]*dUsNew[0:-1,0:-1])/dx
            tempDy = (im1IterNext[dy:,0:-1]*dVsNew[dy:,0:-1]-im1IterNext[0:-1,0:-1]*dVsNew[0:-1,0:-1])/dy
            im1IterNext[0:-1,0:-1] = im1IterNext[0:-1,0:-1] - (tempDx + tempDy)*dt

        ## Start with initial velocities equal to zero, since image has been displaced
        uInitialStep = np.zeros([im1IterNext.shape[0], im1IterNext.shape[1]], dtype='float32')
        vInitialStep = np.zeros([im1IterNext.shape[0], im1IterNext.shape[1]], dtype='float32')
        finalUAccum = usNew
        finalVAccum = vsNew
    else:
        uInitialStep = usNew
        vInitialStep = vsNew
        finalUAccum = np.zeros([im1IterNext.shape[0], im1IterNext.shape[1]], dtype='float32')
        finalVAccum = np.zeros([im1IterNext.shape[0], im1IterNext.shape[1]], dtype='float32')
          

    return im1IterNext, finalUAccum, finalVAccum, uInitialStep, vInitialStep


def genericPyramidalOpticalFlow(im1, im2, FILTER, mainOFlowAlgoAdapter, pyramidalLevels=1, kLevels=1,
                                optionalOFlowAlgoAdapter=None, warping=True, biLinear=True, pyramidalScaling=False):
    """
    Implements a generic Pyramidal topology processing for OpticalFlow algorithms.

    :param im1: image at t=0
    :param im2: image at t=1
    :param FILTER: Gussian kernel filter size
    :param mainOFlowAlgoAdapter: adapter object for the main Optical Flow algorithm 
    :param optionalOFlowAlgoAdapter: adapter object for the optional enhancement Optical Flow algorithm (optional)
    :param pyramidalLevels: number of pyramidalLevels to consider (optional)
    :param kLevels: the number of iterations to perform at each pyramidal level (optional), typically 1 or 2
    :param warping: Apply warping to the image, if True (can be overriden by mainOFlowAlgoAdapter)
    :param biLinear: Apply Bi-Linear interpolation to compute sub-pixel warping, if True, or
                     apply optical flow equation to compute sub-pixel warping, if False (can be overriden by mainOFlowAlgoAdapter)
    :param pyramidalScaling: Set to true to apply velocity scaling when changing pyramidal level (can be overriden by mainOFlowAlgoAdapter)
    :return the estimated U,V velocity components

    The Optical Flow algorithm adapter objects are expected to have the following method signatures:
    
    Method compute(...)
        U, V, error = compute(im1, im2, U, V)
    Inputs:
        im1: image at t=0
        im2: image at t=1
        U: the x-component initial velocities 
        V: the y-component initial velocities        
    Outputs:
        U: the x-component estimated velocities 
        V: the y-component estimated velocities
        error: the final estimated error for the image registration process    

    Method getAlgoName()
        name = getAlgoName()
    Inputs:
        none
    Outputs:
        a string with the algorithm name
        
    Method hasGenericPyramidalDefaults()
        True / False = hasGenericPyramidalDefaults()
    Inputs:
        none
    Outputs:
        True or False if algorithm implementation provides default configurations for GenericPyramidalOpticalFlow
        
    Method getGenericPyramidalDefaults()
        configurationMap = getGenericPyramidalDefaults()
    Inputs:
        none
    Ouputs:
        a configuration map in the form of a dictionary containing any of the following keys: 'warping', 'biLinear' or 'scaling' 
    """

    scale = 1.0/(2.0**(pyramidalLevels-1));
 
    im1IterNew = None 
    im2IterNew = None
    im1IterFiltered = None
    im2IterFiltered = None

    U = None
    V = None
    Uaccum = None
    Vaccum = None
    
    if mainOFlowAlgoAdapter.hasGenericPyramidalDefaults():
        #Perform parameter overriding according to algorithm defaults
        defaultsMap = mainOFlowAlgoAdapter.getGenericPyramidalDefaults()
        if defaultsMap is not None:
            warpingDefault = defaultsMap.get('warping')
            biLinearDefault = defaultsMap.get('biLinear')
            pyramidalScalingDefault = defaultsMap.get('scaling')
            if warpingDefault is not None:
                print('Using algorithm ' + mainOFlowAlgoAdapter.getAlgoName() + 
                      ' default value for warping parameter:', warpingDefault)
                warping = warpingDefault
            if biLinearDefault is not None:
                print('Using algorithm ' + mainOFlowAlgoAdapter.getAlgoName() + 
                      ' default value for biLinear parameter:', biLinearDefault)
                biLinear = biLinearDefault
            if pyramidalScalingDefault is not None:
                print('Using algorithm ' + mainOFlowAlgoAdapter.getAlgoName() + 
                      ' default value for pyramidalScaling parameter:', pyramidalScalingDefault)
                pyramidalScaling = pyramidalScalingDefault

    for level in np.arange(1, pyramidalLevels+1):
        #debug()
        im1IterPrev = im1IterFiltered
             
        if scale < 1.0:
            #The image resizing algorithm plays an important role in final accuracy
            im1IterNew = imresize(im1,
                             (np.int(np.round(np.size(im1,1)*scale)),
                              np.int(np.round(np.size(im1,0)*scale))))
            im2IterNew = imresize(im2,
                             (np.int(np.round(np.size(im1,1)*scale)),
                              np.int(np.round(np.size(im1,0)*scale))))
        elif scale > 1.0:
            raise Exception('Invalid scale level: ' + str(scale))
        else:
            im1IterNew = im1;
            im2IterNew = im2;
            
        if FILTER > 1:
            std = FILTER * 0.62;
            #Radius of the gaussian_filter1d, called by gaussian_filter: lw = int(truncate * sd + 0.5)
            #and in _gaussian_kernel1d(...) -> x = numpy.arange(-radius, radius+1)
            #so d = 2*int(truncate * sd + 0.5) + 1
            #(https://github.com/scipy/scipy/blob/1d6a0b6750a674a6d17abf42a6cc12cb694b7501/scipy/ndimage/filters.py#L168)
            im1IterFiltered = gaussian_filter(im1IterNew,FILTER,truncate=0.7406)
            im2IterFiltered = gaussian_filter(im2IterNew,FILTER,truncate=0.7406)
        else:
            im1IterFiltered = np.copy(im1IterNew)
            im2IterFiltered = im2IterNew

        if level > 1:
            im1IterWork, Uaccum, Vaccum, U, V = updateNextPyramidalLevel(im1IterFiltered, im1IterPrev, 
                                                                              Uaccum, Vaccum, U, V, warping, biLinear, pyramidalScaling);
        else:
            im1IterWork = im1IterFiltered
            # set up initial velocities
            U = np.zeros([im1IterFiltered.shape[0], im1IterFiltered.shape[1]], dtype='float32')
            V = np.zeros([im1IterFiltered.shape[0], im1IterFiltered.shape[1]], dtype='float32')

            #Setup initial accumulated velocities
            Uaccum = np.zeros([im1IterFiltered.shape[0], im1IterFiltered.shape[1]], dtype='float32')
            Vaccum = np.zeros([im1IterFiltered.shape[0], im1IterFiltered.shape[1]], dtype='float32')


        for k in np.arange(0,kLevels):
            print('Level=',level,' kIter=',k)

            if k > 0:
                im1IterWork, Uaccum, Vaccum, U, V = updateNextPyramidalLevel(np.copy(im1IterFiltered), im1IterFiltered,
                                                                                     Uaccum, Vaccum, U, V, warping, biLinear, False);

            U, V, error = mainOFlowAlgoAdapter.compute(im1IterFiltered, im2IterFiltered, U, V)
            print(mainOFlowAlgoAdapter.getAlgoName() + ' estimated error for image registration: ' + str(error));

            if not optionalOFlowAlgoAdapter is None:
                U, V, errorOptional = optionalOFlowAlgoAdapter.compute(im1IterFiltered, im2IterFiltered, U, V)
                print(optionalOFlowAlgoAdapter.getAlgoName() + ' estimated error for image registration: ' + str(errorOptional));

            Uaccum += U
            Vaccum += V
        scale*=2
    return Uaccum,Vaccum

