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
//    Sen Liu, swjtuls1987@126.com
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

import os
from GaussianKernelBitExact import *
import PIL
from PIL import Image
import pyopencl as cl
import pyopencl.array as cl_array
#from IPython.core import debugger
#debug = debugger.Pdb().set_trace

def imresize(im, res):
    return np.array(Image.fromarray(im).resize(res, PIL.Image.BILINEAR))

class Farneback_PyCL(object):
    """
    Provides a port of the OpenCV Farneback OpenCL implementation for Python using PyOpenCL.
    It also implements the GenericPyramidalOpticalFlow client algorithms interface.
    """
    def __init__(self, windowSize=33, Niters=200, polyN=7, polySigma=1.5, useGaussian=True, pyrScale=0.5, pyramidalLevels=1, platformID=0, deviceID=0, \
                       provideGenericPyramidalDefaults=True):
        assert(pyramidalLevels != 1, 'Currently pyramidal levels are not supported through this implementation port')
        self.useDouble = False    #Gains are so marginal that it doesn't justify performance impact
        self.windowSize = windowSize
        self.numIters = Niters
        self.polyN = np.int(polyN)
        self.polySigma = polySigma
        self.useGaussianFilter = useGaussian
        self.pyramidalLevels = pyramidalLevels-1
        self.fastPyramids = False #Currently unimplemented, so always false
        self.pyrScale = pyrScale
        
        self.provideGenericPyramidalDefaults=provideGenericPyramidalDefaults
        
        self.kernelGaussianBlur = None
        self.kernelGaussianBlur5 = None
        self.kernelBoxFilter5 = None
        self.kernelPolyExpansion = None
        self.kernelUpdateFlow = None
        self.kernelUpdateMatrices = None
        
        self.scriptDir = os.path.dirname(os.path.realpath(__file__))

        if windowSize & 1 == 0:
            raise Exception('windowSize must be an odd value')

        dev = cl.get_platforms()[platformID].get_devices()[deviceID]
        ctx = cl.Context([dev])
        print('Runnning on device: ' + str(dev))
        
        #Compile kernels
        clSrc=None
        with open(self.scriptDir + os.path.sep + 'optical_flow_farneback.cl', 'r') as file:
            clSrc = file.read()

        if clSrc == None:
            raise Exception('Couldn''t read optical_flow_farneback.cl')

        self.ctx = ctx
        self.clProg = None
        if self.useDouble:
            self.clProg = cl.Program(ctx, clSrc).build(['-DpolyN=' + str(polyN), '-DpolyN_', '-DUSE_DOUBLE'])
        else:
            self.clProg = cl.Program(ctx, clSrc).build(['-DpolyN=' + str(polyN), '-DpolyN_'])

        self.kernelPolyExpansion = self.clProg.polynomialExpansion
        self.kernelGaussianBlur = self.clProg.gaussianBlur
        self.kernelGaussianBlur5 = self.clProg.gaussianBlur5
        self.kernelUpdateMatrices = self.clProg.updateMatrices
        self.kernelBoxFilter5 = self.clProg.boxFilter5
        self.kernelUpdateFlow = self.clProg.updateFlow

    def FarnebackPrepareGaussian(self):
        n = self.polyN
        sigma = self.polySigma 
        
        if sigma < 1.19209289550781250000000000000000000e-7:
            sigma = n*0.3;
            
        g = np.zeros([2*n+1], dtype=np.float32)
        xg = np.zeros([2*n+1], dtype=np.float32)
        xxg = np.zeros([2*n+1], dtype=np.float32)
        
        s = np.float64(0.0)
        for x in range(-n, n+1):
            g[x + n] = np.exp(-x*x/(2*sigma*sigma));
            s += g[x + n];

        s = 1.0/s;
        for x in range(-n, n+1):
            g[x + n] = np.float32(g[x + n]*s);
            xg[x + n] = np.float32(x*g[x + n]);
            xxg[x + n] = np.float32(x*x*g[x + n]);

        G = np.zeros((6, 6), np.float64);
        for y in range(-n, n+1):
            for x in range(-n, n+1):
                G[0,0] += g[y + n]*g[x + n];
                G[1,1] += g[y + n]*g[x + n]*x*x;
                G[3,3] += g[y + n]*g[x + n]*x*x*x*x;
                G[5,5] += g[y + n]*g[x + n]*x*x*y*y;

        G[2,2] = G[0,3] = G[0,4] = G[3,0] = G[4,0] = G[1,1];
        G[4,4] = G[3,3];
        G[3,4] = G[4,3] = G[5,5];

        #invG:
        #[ x        e  e    ]
        #[    y             ]
        #[       y          ]
        #[ e        z       ]
        #[ e           z    ]
        #[                u ]
        invG = np.linalg.inv(G)

        ig11 = invG[1,1];
        ig03 = invG[0,3];
        ig33 = invG[3,3];
        ig55 = invG[5,5];

        return g, xg, xxg, ig11, ig03, ig33, ig55

    def setPolynomialExpansionConsts(self):
        n = self.polyN 
        m_igd = np.zeros(4, dtype=np.float64)
        m_ig = np.zeros(4, dtype=np.float32)
        g, xg, xxg, m_igd[0], m_igd[1], m_igd[2], m_igd[3] = self.FarnebackPrepareGaussian();

        t_g = g[n:].reshape([1, n+1]);
        t_xg = xg[n:].reshape([1, n+1]);
        t_xxg = xxg[n:].reshape([1, n+1]);
        
        m_g = t_g.copy();
        m_xg = t_xg.copy();
        m_xxg = t_xxg.copy();

        m_ig[0] = np.float32(m_igd[0]);
        m_ig[1] = np.float32(m_igd[1]);
        m_ig[2] = np.float32(m_igd[2]);
        m_ig[3] = np.float32(m_igd[3]);
        
        self.matrixG = m_g;
        self.matrixXG = m_xg;
        self.matrixXXG = m_xxg;
        self.matrixIG = m_ig;
        self.matrixIGD = m_igd;

    def getGaussianKernel(self, n, sigma):
        _, kernel_bitexact = getGaussianKernelBitExact(n, sigma);
        return np.float32(kernel_bitexact).reshape([1, n])

    def setGaussianBlurKernel(self, smoothSize, sigma):
        g = self.getGaussianKernel(smoothSize, sigma)
        #print(g, np.sum(g))
        m_gKer = np.zeros([1, np.int(smoothSize/2) + 1], dtype=np.float32)
        m_gKer[0,:] = g[0, np.int(smoothSize/2):]
        #print(m_gKer)
        self.matrixGKernel = m_gKer

    def gaussianBlur(self, src, kSizeHalf, dst):
        #Matrices must be in row order
        srcRows = src.shape[0]
        srcCols = src.shape[1]

        dstRows = cl.cltypes.int(dst.shape[0])
        dstCols = cl.cltypes.int(dst.shape[1])

        numChannels   = 1
        workGroupSize = (256, 1)
        globalSize    = (srcCols, srcRows)

        smemSize = np.int((workGroupSize[0] + 2*kSizeHalf) * np.dtype(np.float32).itemsize)

        mf = cl.mem_flags
        srcCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=src)
        gaussianMatKernelCl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.matrixGKernel)
        dstCl = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=dst)
        
        queue = cl.CommandQueue(self.ctx)
        self.kernelGaussianBlur(queue, globalSize, workGroupSize, 
                           srcCl, cl.cltypes.int(srcCols*numChannels), dstCl, cl.cltypes.int(dstCols*numChannels), 
                           dstRows, dstCols, gaussianMatKernelCl, cl.cltypes.int(kSizeHalf), cl.LocalMemory(smemSize))

        cl.enqueue_copy(queue, dst, dstCl)
        
        if dst is None:
            raise Exception('Failed to compute gaussianBlur')

        return dst

    def gaussianBlur5(self, src, kSizeHalf, dst):
        #Matrices must be in row order
        srcRows = cl.cltypes.int(src.shape[0])
        srcCols = cl.cltypes.int(src.shape[1])

        dstRows = cl.cltypes.int(dst.shape[0])
        dstCols = cl.cltypes.int(dst.shape[1])

        height = np.int(srcRows / 5)
        numChannels   = cl.cltypes.int(1)
        workGroupSize = (256, 1)
        globalSize    = (srcCols, height)

        smemSize = np.int((workGroupSize[0] + 2*kSizeHalf) * 5 * np.dtype(np.float32).itemsize)

        mf = cl.mem_flags
        srcCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=src)
        gaussianMatKernelCl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.matrixGKernel)
        dstCl = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=dst)
        
        queue = cl.CommandQueue(self.ctx)
        self.kernelGaussianBlur5(queue, globalSize, workGroupSize, 
                           srcCl, srcCols*numChannels, dstCl, dstCols*numChannels, 
                           cl.cltypes.int(height), srcCols, gaussianMatKernelCl, cl.cltypes.int(kSizeHalf), cl.LocalMemory(smemSize))

        cl.enqueue_copy(queue, dst, dstCl)

        if dst is None:
            raise Exception('Failed to compute gaussianBlur5')

        return dst

    def boxFilter5(self, src, kSizeHalf, dst):
        #Matrices must be in row order
        srcRows       = cl.cltypes.int(src.shape[0])
        srcCols       = cl.cltypes.int(src.shape[1])

        dstRows       = cl.cltypes.int(dst.shape[0])
        dstCols       = cl.cltypes.int(dst.shape[1])

        height        = np.int(srcRows / 5)
        workGroupSize = (256, 1)
        globalSize    = (srcCols, height)
        numChannels   = cl.cltypes.int(1)

        smemSize = np.int((workGroupSize[0] + 2*kSizeHalf) * 5 * np.dtype(np.float32).itemsize)

        #Step0 is the size of a row in bytes (columns * number of channels * dataTypeSize)
        #step0  = np.dtype(np.float32).itemsize * src.shape[1]
        #Step1 is the size of an element in bytes
        #step1  = np.dtype(np.float32).itemsize
        #flowx.step / flowx.elemSize() is just then number of columns * number of channels
        
        mf = cl.mem_flags
        srcCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=src)
        dstCl = cl.Buffer(self.ctx, mf.WRITE_ONLY  | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=dst)

        queue = cl.CommandQueue(self.ctx)
        self.kernelBoxFilter5(queue, globalSize, workGroupSize, 
                              srcCl, srcCols*numChannels, dstCl, dstCols*numChannels,
                              cl.cltypes.int(height), srcCols, cl.cltypes.int(kSizeHalf), cl.LocalMemory(smemSize))

        cl.enqueue_copy(queue, dst, dstCl)
        
        if dst is None:
            raise Exception('Failed to compute boxFilter5')

        return dst        

    def polynomialExpansion(self, src, dst):
        #Matrices must be in row order
        srcRows = cl.cltypes.int(src.shape[0])
        srcCols = cl.cltypes.int(src.shape[1])
        dstRows = cl.cltypes.int(dst.shape[0])
        dstCols = cl.cltypes.int(dst.shape[1])
        numChannels = cl.cltypes.int(1)
        workGroupSize  = (256,1)
        globalSize = (np.int((srcCols + workGroupSize[0] - 2*self.polyN - 1) / (workGroupSize[0] - 2*self.polyN)) * workGroupSize[0], \
                      srcRows)

        smemSize = np.int(3 * workGroupSize[0] * np.dtype(np.float32).itemsize)

        mf = cl.mem_flags
        srcCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=src)
        dstCl = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=dst)
        gCl   = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=self.matrixG)
        xgCl  = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=self.matrixXG)
        xxgCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=self.matrixXXG)
        #igCl  = cl.Buffer(self.ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=self.matrixIG)

        igCl = None
        if self.useDouble:
            igCl = np.zeros(1, dtype=cl_array.vec.double4)
            igCl['x'] = self.matrixIGD[0]
            igCl['y'] = self.matrixIGD[1]
            igCl['z'] = self.matrixIGD[2]
            igCl['w'] = self.matrixIGD[3]
        else:
            igCl = np.zeros(1, dtype=cl_array.vec.float4)
            igCl['x'] = self.matrixIG[0]
            igCl['y'] = self.matrixIG[1]
            igCl['z'] = self.matrixIG[2]
            igCl['w'] = self.matrixIG[3]

        queue = cl.CommandQueue(self.ctx)
        self.kernelPolyExpansion(queue, globalSize, workGroupSize,
                                 srcCl, srcCols * numChannels, dstCl, dstCols * numChannels, 
                                 srcRows, srcCols, gCl, xgCl, xxgCl, cl.LocalMemory(smemSize), igCl)

        cl.enqueue_copy(queue, dst, dstCl)

        if dst is None:
            raise Exception('Failed to compute polynomial expansion')

        return dst

    def updateFlow(self, M, flowX, flowY):
        #Matrices must be in row order
        flowXRows = cl.cltypes.int(flowX.shape[0])
        flowXCols = cl.cltypes.int(flowX.shape[1])

        flowYRows = cl.cltypes.int(flowY.shape[0])
        flowYCols = cl.cltypes.int(flowY.shape[1])

        numChannels = cl.cltypes.int(1)
        workGroup   = (32, 8)
        globalSize  = (flowXCols, flowXRows)

        mf = cl.mem_flags
        MCl     = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=M)
        flowXCl = cl.Buffer(self.ctx, mf.WRITE_ONLY  | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=flowX)
        flowYCl = cl.Buffer(self.ctx, mf.WRITE_ONLY  | mf.ALLOC_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=flowY)

        queue = cl.CommandQueue(self.ctx)
        self.kernelUpdateFlow(queue, globalSize, workGroup, 
                              MCl, cl.cltypes.int(M.shape[1]*numChannels), 
                              flowXCl, flowXCols*numChannels, flowYCl, flowYCols*numChannels,
                              flowYRows, flowYCols)

        cl.enqueue_copy(queue, flowX, flowXCl)
        cl.enqueue_copy(queue, flowY, flowYCl)

        if flowX is None:
            raise Exception('Failed to updateFlow')
        if flowY is None:
            raise Exception('Failed to updateFlow')

        return flowX, flowY

    def updateMatrices(self, flowX, flowY, RA, RB, M):
        #Matrices must be in row order
        MCols      = cl.cltypes.int(M.shape[1])

        flowXRows  = cl.cltypes.int(flowX.shape[0])
        flowXCols  = cl.cltypes.int(flowX.shape[1])

        flowYCols  = cl.cltypes.int(flowY.shape[1])

        numChannels = cl.cltypes.int(1)
        workGroup   = (32, 8)
        globalSize  = (flowXCols, flowXRows)
        
        mf = cl.mem_flags
        flowXCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=flowX)
        flowYCl = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=flowY)
        RACl    = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=RA)
        RBCl    = cl.Buffer(self.ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=RB)
        MCl     = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=M)

        queue = cl.CommandQueue(self.ctx)
        self.kernelUpdateMatrices(queue, globalSize, workGroup, 
                                  flowXCl, flowXCols * numChannels, flowYCl, flowYCols*numChannels,
                                  flowXRows, flowXCols, 
                                  RACl, cl.cltypes.int(RA.shape[1]*numChannels), 
                                  RBCl, cl.cltypes.int(RB.shape[1]*numChannels), MCl, MCols*numChannels)
                                               
        cl.enqueue_copy(queue, M, MCl)

        if M is None:
            raise Exception('Failed to updateMatrices')

        return M

    def updateFlowBoxFilter(self, RA, RB, flowX, flowY, M, bufM, blockSize, updateMatrices):
        try:
            bufM = self.boxFilter5(M, blockSize/2, bufM)
        except Exception as e:
            raise Exception('Failed to compute boxFilter5 in updateFlowBoxFilter', e)

        #swap(M, bufM)
        M, bufM = bufM, M
        try:
            flowX, flowY = self.updateFlow(M, flowX, flowY)
        except Exception as e:
            raise Exception('Failed to updateFlow in updateFlowBoxFilter', e)
        if updateMatrices:
            M = self.updateMatrices(flowX, flowY, RA, RB, M)
            
        return flowX, flowY, M, bufM

    def updateFlowGaussianBlur(self, RA, RB, flowX, flowY, M, bufM, blockSize, updateMatrices):
        try:
            bufM = self.gaussianBlur5(M, blockSize/2, bufM)
        except Exception as e:
            raise Exception('Failed to compute gaussianBlur5 in updateFlowGaussianBlur', e)
            
        #swap(M, bufM)
        M, bufM = bufM, M
        try:
            flowX, flowY = self.updateFlow(M, flowX, flowY)
        except Exception as e:
            raise Exception('Failed to updateFlow in upateFlowGaussianBlur', e)
        if updateMatrices:
            M = self.updateMatrices(flowX, flowY, RA, RB, M)

        return flowX, flowY, M, bufM

    def pyrDown(self, pyramid):
        raise Exception('Fast Pyramids not supported yet... Check ocl_pyrDown and ocl_pyrUp, pyrDown, pyrUp in OpenCV')

    def compute(self, im1, im2, U, V):
        assert(self.polyN == 5 or self.polyN == 7) 
        assert(im1.shape == im2.shape and self.pyrScale < 1)
        assert(U.shape == im1.shape and V.shape == im1.shape)
        assert(not self.fastPyramids or abs(self.pyrScale - 0.5) < 1e-6)

        min_size = 32;

        size = im1.shape; #[1] - width, [0] - height
        prevFlowX=None
        prevFlowY=None
        curFlowX=None
        curFlowY=None
        
        flowX0 = U
        flowY0 = V

        #Crop unnecessary pyramidal levels
        scale = 1;
        finalNumLevels = 0;
        while finalNumLevels < self.pyramidalLevels:         
            scale *= self.pyrScale;
            if (size[1]*scale < min_size or size[0]*scale < min_size):
                break;
            finalNumLevels+=1


        if (self.fastPyramids):
            # Build Gaussian pyramids using pyrDown()
            #Will have finalNumLevels + 1 elements
            pyramid0 = []
            pyramid1 = []
            pyramid0.append(im1);
            pyramid1.append(im2);
            
            i = 1
            curPyramid0 = pyramid0[0]
            curPyramid1 = pyramid1[0]
            while i <= finalNumLevels:
                curPyramid0 = pyrDown(curPyramid0)
                curPyramid1 = pyrDown(curPyramid1)
                pyramid0.append(curPyramid0)
                pyramid1.append(curPyramid1)

        self.setPolynomialExpansionConsts()

        for k in np.arange(finalNumLevels, 0 - 1, -1):
            scale = 1.0
            for i in np.arange(0, k):
                scale *= self.pyrScale
        
            sigma = (1.0/scale - 1.0) * 0.5
            smoothSize = np.int(round(sigma*5)) | 1
            smoothSize = max(smoothSize, 3)
        
            width  = np.int(round(size[1] * scale))
            height = np.int(round(size[0] * scale))
         
            if self.fastPyramids:            
                width = pyramid0[k].shape[1];
                height = pyramid0[k].shape[0];
        
            #if (k > 0):
            #    curFlowX = np.zeros([height, width], dtype=np.float32)
            #    curFlowY = np.zeros([height, width], dtype=np.float32)                
            #else:
            #    curFlowX = flowX0;
            #    curFlowY = flowY0;
        
            if prevFlowX is None:
                curFlowX = imresize(flowX0, (width, height))
                curFlowY = imresize(flowY0, (width, height))                
                curFlowX *= scale
                curFlowY *= scale
            else:
                curFlowX = imresize(prevFlowX, (width, height))
                curFlowY = imresize(prevFlowY, (width, height))                
                curFlowX *= 1.0/self.pyrScale
                curFlowY *= 1.0/self.pyrScale

            M    = np.zeros((5*height, width), np.float32); 
            bufM = np.zeros((5*height, width), np.float32);
            RA   = np.zeros((5*height, width), np.float32); 
            RB   = np.zeros((5*height, width), np.float32);

            if self.fastPyramids:
                try:
                    RA = polynomialExpansion(pyramid0[k], RA)
                except Exception as e:
                    raise Exception('Failed to do polynomial expansion for RA', e)
                try:
                    RB = polynomialExpansion(pyramid1[k], RB)
                except Exception as e:
                    raise Exception('Failed to do polynomial expansion for RB', e)
            else:
                blurredFrameA = np.zeros([size[0], size[1]], dtype=np.float32)
                blurredFrameB = np.zeros([size[0], size[1]], dtype=np.float32)                

                pyrLevelA = np.zeros([height, width], dtype=np.float32)
                pyrLevelB = np.zeros([height, width], dtype=np.float32)                

                self.setGaussianBlurKernel(smoothSize, sigma);

                #Frame A
                try:
                    blurredFrameA = self.gaussianBlur(im1, smoothSize/2, blurredFrameA)
                except Exception as e:
                    raise Exception('Failed to do gaussian Blur for frame A, level ' + str(i), e)
                pyrLevelA = imresize(blurredFrameA, (width, height))
                try:                
                    RA = self.polynomialExpansion(pyrLevelA, RA)
                except Exception as e:
                    raise Exception('Failed to do polynomial expansion for frame A, level ' + str(i), e)

                #Frame B
                try:
                    blurredFrameB = self.gaussianBlur(im2, smoothSize/2, blurredFrameB)
                except Exception as e:
                    raise Exception('Failed to do gaussian Blur for frame B, level ' + str(i), e)
                pyrLevelB = imresize(blurredFrameB, (width, height))
                try:                
                    RB = self.polynomialExpansion(pyrLevelB, RB)
                except Exception as e:
                    raise Exception('Failed to do polynomial expansion for frame A, level ' + str(i), e)
            
            M = self.updateMatrices(curFlowX, curFlowY, RA, RB, M)           

            if self.useGaussianFilter:
                self.setGaussianBlurKernel(self.windowSize, self.windowSize/2*0.3)
            for  i in np.arange(0, self.numIters):
                if self.useGaussianFilter:                    
                    curFlowX, curFlowY, M, bufM = self.updateFlowGaussianBlur(RA, RB, curFlowX, curFlowY, M, bufM, self.windowSize, i < self.numIters-1)
                else:
                    curFlowX, curFlowY, M, bufM = self.updateFlowBoxFilter(RA, RB, curFlowX, curFlowY, M, bufM, self.windowSize, i < self.numIters-1)

            prevFlowX = curFlowX
            prevFlowY = curFlowY

        U = curFlowX
        V = curFlowY
        error = 'Unknown'

        return U, V, error

    def getAlgoName(self):
         return 'Farneback CL'

    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults
        
    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = False
        parameters['scaling'] = False
        return parameters    
