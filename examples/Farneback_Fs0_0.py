"""
MIT LicenseCopyright (c) [2021] [Luís Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt]

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

import sys
import os
sys.path.insert(0, os.path.join('..','src'))
import numpy as np
from skimage.io import imread
import scipy.io
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter
from Farneback_PyCL import Farneback_PyCL #Pure Python and OpenCL code (implementation from OpenCV)

def save_flow(U, V, filename):
    margins = { 'top' : 0,
                'left' : 0 }

    results = { 'u' : U,
                'v' : V,
                'iaWidth' : 1,
                'iaHeight' : 1,
                'margins' : margins  }

    parameters = { 'overlapFactor' : 1.0,
                   'imageHeight' : np.size(U, 0),
                   'imageWidth' : np.size(U, 1) }
    
    scipy.io.savemat(filename, mdict={'velocities': results, 'parameters': parameters}) 

FILTER=0
pyramidalLevels = 1
kLevels = 1
useLiuShenOF = False
basePath=os.path.join('testImages','Bits08','Ni06')
fn1=os.path.join(basePath, 'parabolic01_0.tif')
fn2=os.path.join(basePath, 'parabolic01_1.tif')

print(fn1);

#Iold = imread(fn1,flatten=True).astype(np.float32)  #flatten=True is rgb2gray
#Inew = imread(fn2,flatten=True).astype(np.float32)
#Iold = imread(fn1,as_gray=True).astype(np.float32)  #as_gray=True only supported after Ubuntu 18.04 
#Inew = imread(fn2,as_gray=True).astype(np.float32) 
Iold = imread(fn1).astype(np.float32)
Inew = imread(fn2).astype(np.float32)

#fAdapter = FarnebackOpticalFlowAdapter(pyramidalLevels = 0)
fAdapter = Farneback_PyCL(platformID=0)
if useLiuShenOF:
    lsAdapter = LiuShenOpticalFlowAlgoAdapter(100000000000)
else:
    lsAdapter = None

[U,V] = genericPyramidalOpticalFlow(Iold, Inew, FILTER, fAdapter, pyramidalLevels, kLevels, lsAdapter)

save_flow(U, V, os.path.join('.', 'Farneback_Fs0_0.mat'))

