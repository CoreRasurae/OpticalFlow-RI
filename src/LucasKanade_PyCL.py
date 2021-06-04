#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
//    Yao Wang, bitwangyaoyao@gmail.com
//    Xiaopeng Fu, fuxiaopeng2222@163.com
//
// Ported to Python by: Luis Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt
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

from __future__ import absolute_import, print_function
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import copy

class LucasKanade_PyCl(object):
    def __init__(self, platformID=0, deviceID=0, Niter=600, halfWindow=10, provideGenericPyramidalDefaults=True):
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults
        
        scriptDir = os.path.dirname(os.path.realpath(__file__))

        dev = cl.get_platforms()[platformID].get_devices()[deviceID]
        ctx = cl.Context([dev])
        print('Runnning on device: ' + str(dev))

        if not dev.get_info(cl.device_info.IMAGE_SUPPORT):
             raise Exception('OPENCL Device does not support IMAGEs')

        self.windowHalfWidth = cl.cltypes.int(halfWindow)
        self.windowHalfHeight= cl.cltypes.int(halfWindow)
        self.windowWidth  = cl.cltypes.int(2*halfWindow + 1)
        self.windowHeight = cl.cltypes.int(2*halfWindow + 1)
        self.Niter = Niter

        wsx = 1
        wsy = 1
        if self.windowWidth < 16:
            wsx = 0
        if self.windowHeight < 16:
            wsy = 0

        clSrc=None
        with open(scriptDir + os.path.sep + 'pyrlk.cl', 'r') as file:
            clSrc = file.read()

        if clSrc == None:
            raise Exception('Couldn''t read pyrlk.cl')

        self.clProg = cl.Program(ctx, clSrc).build(['-DWSX='+str(wsx), '-DWSY='+str(wsy)])
        self.ctx = ctx

    def calcPatchSize(self, windowWidth, windowHeight):
        blockX = 0
        blockY = 0
        blockZ = 0

        if windowWidth > 32 and windowWidth > 2 * windowHeight:
            blockX = 32;
            blockY = 8;
        else:
            blockX = 16;
            blockY = 16;

        patchX = np.float32((windowWidth  + blockX - 1) / blockX);
        patchY = np.float32((windowHeight + blockY - 1) / blockY);

        blockZ = patchZ = 1;

        return patchX, patchY

    def compute(self, im1, im2, U, V):
        level = cl.cltypes.int(0) #Number of pyramidal levels
        dimI = cl.cltypes.int(im1.shape[0])
        dimJ = cl.cltypes.int(im1.shape[1])

        workGroup=(8, 8)
        globalThreads=(8 * dimI*dimJ, 8)

        calcErr = False
        if level == 0:
            calcErr = True

        windowWidth = self.windowWidth
        windowHeight= self.windowHeight

        patchX, patchY = self.calcPatchSize(windowWidth, windowHeight)

        im1Border = im1
        im2Border = im2
        
        im1Cl = cl.image_from_array(self.ctx, im1Border, num_channels=1, mode='r')
        im2Cl = cl.image_from_array(self.ctx, im2Border, num_channels=1, mode='r')

        prevPts = np.zeros(dimI * dimJ, dtype=cl.cltypes.float2)
        nextPts = np.zeros(dimI * dimJ, dtype=cl.cltypes.float2)
        status  = np.ones(dimI * dimJ, dtype=cl.cltypes.float)
        err     = np.zeros(dimI * dimJ, dtype=cl.cltypes.uint8)

        i = 0
        while i < dimI:
            j = 0
            while j < dimJ:
                prevPts[i * dimJ + j]=(j,i)
                nextPts[i * dimJ + j]=(j/2.0+U[i,j]/2.0,i/2.0+V[i,j]/2.0)
                j+=1
            i+=1

        mf = cl.mem_flags
        prevPtsCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=prevPts)
        nextPtsCl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=nextPts)
        statusCl  = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=status)
        errCl     = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=err)

        queue = cl.CommandQueue(self.ctx)
        lkSparseKernel = self.clProg.lkSparse
        lkSparseKernel(queue, globalThreads, workGroup, 
                              im1Cl, im2Cl, prevPtsCl, nextPtsCl, statusCl, errCl,
                              level, dimI, dimJ, patchX, patchY, windowWidth, windowHeight,
                              cl.cltypes.int(self.Niter), cl.cltypes.char(calcErr))
        cl.enqueue_copy(queue, nextPts, nextPtsCl)
        cl.enqueue_copy(queue, status, statusCl)
        cl.enqueue_copy(queue, err, errCl)
        
        i = 0
        while i < dimI:
            j = 0
            while j < dimJ:                
                #But it seems that the error is even less when considering all points
                U[i,j] = nextPts[i * dimJ + j][0]-prevPts[i * dimJ + j][0]
                V[i,j] = nextPts[i * dimJ + j][1]-prevPts[i * dimJ + j][1]
                j+=1
            i+=1
        return U,V,calcErr
        
    def getAlgoName(self):
         return 'OpenCL LK'

    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults
        
    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = False
        parameters['scaling'] = False
        return parameters    
