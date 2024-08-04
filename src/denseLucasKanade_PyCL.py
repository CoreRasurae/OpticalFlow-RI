"""
MIT LicenseCopyright (c) [2021-2024] [Luís Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt]

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:The above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import os
import numpy as np
from scipy.ndimage.filters import convolve as filter2
import pyopencl as cl
import pyopencl.array as cl_array
import copy

class denseLucasKanade_PyCl(object):
    def __init__(self, platformID=0, deviceID=0, Niter=5, halfWindow=13, provideGenericPyramidalDefaults=True, enableVorticityEnhancement=False):
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults
        self.enableVorticityEnhancement = enableVorticityEnhancement
        
        scriptDir = os.path.dirname(os.path.realpath(__file__))

        dev = cl.get_platforms()[platformID].get_devices()[deviceID]
        ctx = cl.Context([dev])
        print('Runnning on device: ' + str(dev))
        #pyOpenCL can't retrieve this value
        #print(dev.get_info('CL_DEVICE_IMAGE_PITCH_ALIGNMENT'))
        #AMD RX5700 has a pitch alignment of 256 so the matrix columns must be a multiple of the pitch alignment

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
        with open(scriptDir + os.path.sep + 'pyrlkDenseLargeW.cl', 'r') as file:
            clSrc = file.read()

        if clSrc == None:
            raise Exception('Couldn''t read pyrlk.cl')

        self.clProg = cl.Program(ctx, clSrc).build(['-DWSX='+str(wsx), '-DWSY='+str(wsy)])
        #self.clProg = cl.Program(ctx, clSrc).build(['-DCPU'])
        self.ctx = ctx

    def evaluateVorticityEnhancement(self, U, V):
        if not self.enableVorticityEnhancement:
            return [0, 0, 0, 0]
            
        D = np.array([[0, -1, 0],
                      [0,  0, 0],
                      [0,  1, 0]], dtype=np.float32) * np.float32(0.5)
                 
        Dv=filter2(V, D.T, mode='reflect')
        Du=filter2(U, D,  mode='reflect')
        omega = Dv - Du
        if np.mean(omega) < -2e-3:
            #Left,Right,Top,Bottom
            return [0,1,0,1]
        elif np.mean(omega) > 2e-3:
            return [1,0,0,1]
        else:
            return [0,0,0,0]

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

        patchX = np.int32((windowWidth  + blockX - 1) / blockX);
        patchY = np.int32((windowHeight + blockY - 1) / blockY);

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

        assymetricWndCfg = self.evaluateVorticityEnhancement(U, V)

        #This works well, for images of 512x512, beacause the number of cols 
        #is an exact multiple of RX5700 CL_DEVICE_IMAGE_PITCH_ALIGNMENT=256
        im1Border = im1
        im2Border = im2
        
        u = np.zeros(dimI * dimJ, dtype=cl.cltypes.float)
        v = np.zeros(dimI * dimJ, dtype=cl.cltypes.float)
        im1Cl = cl.image_from_array(self.ctx, im1Border, num_channels=1, mode='r')
        im2Cl = cl.image_from_array(self.ctx, im2Border, num_channels=1, mode='r')
        
        u[:] = U.reshape(dimI * dimJ)
        v[:] = V.reshape(dimI * dimJ)

        status  = np.ones(dimI * dimJ, dtype=cl.cltypes.float)
        err     = np.zeros(dimI * dimJ, dtype=cl.cltypes.uint8)

        mf = cl.mem_flags
        uCl       = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
        vCl       = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v)
        statusCl  = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=status)
        errCl     = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=err)

        queue = cl.CommandQueue(self.ctx)
        lkDenseKernel = self.clProg.lkDense
        lkDenseKernel(queue, globalThreads, workGroup, 
                              im1Cl, im2Cl, uCl, vCl, statusCl, errCl,
                              level, dimI, dimJ, patchX, patchY, windowWidth, windowHeight, 
                              cl.cltypes.int(assymetricWndCfg[0]), cl.cltypes.int(assymetricWndCfg[1]), cl.cltypes.int(assymetricWndCfg[2]), cl.cltypes.int(assymetricWndCfg[3]),
                              cl.cltypes.int(self.Niter), cl.cltypes.char(calcErr))
        cl.enqueue_copy(queue, u, uCl)
        cl.enqueue_copy(queue, v, vCl)
        cl.enqueue_copy(queue, status, statusCl)
        cl.enqueue_copy(queue, err, errCl)
        
        U = u.reshape(dimI, dimJ)
        V = v.reshape(dimI, dimJ)
        
        return U,V,calcErr
        
    def getAlgoName(self):
         return 'OpenCL Dense LK'

    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults
        
    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = False
        parameters['intermediateScaling'] = True
        parameters['scaling'] = False
        return parameters
        
