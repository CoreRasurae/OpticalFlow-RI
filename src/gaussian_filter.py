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
import numpy as np
from numba import jit, njit

@njit(nopython=True, cache=True)
def convolveDirect(a, b, result):
   sizeA = a.shape[0]
   sizeB = b.shape[0]
   minSize = min(sizeA, sizeB)
   halfSize = int(minSize/2)
   minVec = a
   otherVec = b
   otherSize = sizeB
   if minSize == sizeB:
      minVec = b
      otherVec = a
      otherSize = sizeA
   for i in range(sizeA + sizeB - 1):
      for j in range(minSize):
         if i - j >= 0 and i - j < otherSize:
            result[i] += otherVec[i - j] * minVec[j]
   if halfSize * 2 == minSize:
      result = result[halfSize-1:otherSize+halfSize-1]
   else:
      result = result[halfSize:halfSize+otherSize]
   return result

def prepareGaussianKernel(sigma, kernelSizePx):
   kernel = np.zeros(kernelSizePx, dtype=np.float32)
   xs = np.arange(-kernelSizePx/2, kernelSizePx/2, 1, dtype=np.int)
   kernel[:] = 1.0/np.sqrt(2.0 * np.pi * sigma**2) * np.exp( -xs**2/(2.0 * sigma**2) )
   kernel /= np.sum(kernel)
   return kernel
   
def convolveSeparableFilter(kernel, image):
   sizeY, sizeX = image.shape
   sizeK = kernel.shape[0]
   halfSize = np.int(sizeK/2)
   tempY = np.zeros([sizeX + sizeK - 1], dtype=np.float32)
   tempX = np.zeros([sizeY + sizeK - 1], dtype=np.float32)
   
   resultY = np.zeros([sizeX + 2*halfSize + sizeK - 1], dtype=np.float32)
   for i in np.arange(sizeY):
      tempY[halfSize:-halfSize] = image[i, :]
      for j in np.arange(halfSize):
         tempY[halfSize - 1 - j] = image[i, j]
         tempY[sizeX + sizeK - 2 - j] = image[i, sizeX - 1 - j]

      #image[i, :] = np.convolve(tempY, kernel, 'same')[halfSize:-halfSize]
      #which is equivalent to:
      image[i, :] = convolveDirect(tempY, kernel, resultY)[halfSize:-halfSize]
      resultY[:] = 0
   
   resultX = np.zeros([sizeY + 2*halfSize + sizeK - 1], dtype=np.float32)   
   for j in np.arange(sizeX):
      tempX[halfSize:-halfSize] = image[:, j]
      for i in np.arange(halfSize):
         tempX[halfSize - 1 - i] = image[i, j]
         tempX[sizeY + sizeK - 2 - i] = image[sizeY - 1 - i, j]
      
      #image[:, j]  = np.convolve(tempX, kernel, 'same')[halfSize:-halfSize]
      #which is equivalent to:
      image[:, j] = convolveDirect(tempX, kernel, resultX)[halfSize:-halfSize]      
      resultX[:] = 0
      
   return image

def gaussian_filter(image, sigma, truncate):
    kernelSizePx = 2*int(truncate * sigma + 0.5) + 1
    kernel = prepareGaussianKernel(sigma, kernelSizePx)
    return convolveSeparableFilter(kernel, image)
    
def gaussian_filterPx(image, sigma, kernelSizePx):
    kernel = prepareGaussianKernel(sigma, kernelSizePx)
    return convolveSeparableFilter(kernel, image)    
