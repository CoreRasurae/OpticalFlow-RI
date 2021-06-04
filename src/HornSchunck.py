#!/usr/bin/env python
"""
MIT License

Copyright (c) 2020 Chinmay Wyawahare
Copyright (c) 2021 Luís Mendes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This implementation was carried out by:
 Luis Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt
and is a derived work from the implementation provided in:
 https://github.com/gandalf1819/Optical-Flow by Chinmay Wyawahare.
"""

from __future__ import division
from numba import jit,njit,prange,objmode
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve as filter2
import scipy.signal as sgn
from scipy.linalg import norm
from scipy import signal
#from IPython.core import debugger
#debug = debugger.Pdb().set_trace
#

class HSOpticalFlowAlgoAdapter(object):
    def __init__(self, alphas, Niter, provideGenericPyramidalDefaults=True):
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults
        self.alphas = alphas
        self.Niter = Niter

    def compute(self, im1, im2, U, V):
        alpha = self.alphas.pop() #Remove last alpha from the list
        return HS(im1, im2, alpha, self.Niter, U, V)

    def getAlgoName(self):
        return 'Horn-Schunck'
        
    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults
        
    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = True
        parameters['biLinear'] = True
        parameters['scaling'] = True
        return parameters        

@jit('Tuple((float32[:,:],float32[:,:]))(float32,float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])', nopython=True, parallel=False)
def HS_helper2(alpha, fx, fy, ft, uAvg, vAvg):
       #%% common part of update step
       der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
       #%% iterative step	
       Unew = uAvg - fx * der
       Vnew = vAvg - fy * der
       return Unew, Vnew

#@njit(parallel=True)
def HS_helper(alpha, Niter, kernel, U, V, fx, fy, ft):
    for _ in np.arange(Niter):
        #%% Compute local averages of the flow vectors
        uAvg = filter2(U,kernel, mode='mirror') #uBar in the paper
        vAvg = filter2(V,kernel, mode='mirror') #vBar in the paper
        
        U, V = HS_helper2(alpha, fx, fy, ft, uAvg, vAvg)
    return U, V

def HS(im2, im1, alpha, Niter, U, V):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    U: initial x-component velocity vector
    V: initial y-component velocity vector
    """

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel=np.array([[1/12, 1/6, 1/12],
                     [1/6,    0, 1/6],
                     [1/12, 1/6, 1/12]],np.float32) #Kernel for uBar and vBar in the paper

    if (np.size(fx,0) > 100 and np.size(fx,1) > 100):
        print(fx[100,100],fy[100,100],ft[100,100])

    total_error=100000000;
    # Iteration to reduce error
    Unew = None
    VNew = None

    Unew, Vnew = HS_helper(alpha, Niter, kernel, U, V, fx, fy, ft)
    total_error = (norm(Unew-U,'fro')+norm(Vnew-V,'fro'))/(im1.shape[0]*im1.shape[1])
    
    U = Unew
    V = Vnew

    return U, V, total_error

def computeDerivatives(im1, im2):
    eMode='mirror' #Convolution extension mode
    #%% build kernels for calculating derivatives
    kernelX = np.array([[-1, 1],
                        [-1, 1]], dtype='float32') * .25 #kernel for computing d/dx
    kernelY = np.array([[-1,-1],
                        [ 1, 1]], dtype='float32') * .25 #kernel for computing d/dy

    
    kernelT = np.ones((2,2), dtype='float32')*.25

    #Ex in the paper (in the centre of 2x2 cube)
    fx = filter2(im1, kernelX, mode=eMode) + filter2(im2, kernelX, mode=eMode)
    #Ey in the paper (in the centre of 2x2 cube)
    fy = filter2(im1, kernelY, mode=eMode) + filter2(im2, kernelY, mode=eMode)

    #ft = im2 - im1
    #Et in the paper (in the centre of 2x2 cube)
    ft = filter2(im2, kernelT, mode=eMode) + filter2(im1,-kernelT, mode=eMode)

    return fx,fy,ft
