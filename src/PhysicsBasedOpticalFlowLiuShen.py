#!/usr/bin/env python
"""
MIT LicenseCopyright (c) [2017] [OpenOpticalFlow]

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

Ported to Python by: Luis Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt

"""

from numba import jit
import numpy as np
from scipy.linalg import norm
from scipy.ndimage.filters import convolve as filter2
from scipy import signal

class LiuShenOpticalFlowAlgoAdapter(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def compute(self, im1, im2, U, V):
        return physicsBasedOpticalFlowLiuShen(im1, im2, self.alpha, U, V)

    def getAlgoName(self):
        return 'Liu-Shen Physics based OF'
        
    def hasGenericPyramidalDefaults():
        return False

def generate_invmatrix(im, h, dx):
    D  = np.array([ [  0, -1,  0], [  0,  0,  0], [  0,  1,  0] ])/2; # partial derivative 
    M  = np.array([ [  1,  0, -1], [  0,  0,  0], [ -1,  0,  1] ])/4; # mixed partial derivatives
    F  = np.array([ [  0,  0,  0], [  0,  1,  1], [  0,  1,  1] ])/4; # average
    D2 = np.array([ [  0,  1,  0], [  0, -2,  0], [  0,  1,  0] ]);   # partial derivative
    H  = np.array([ [  1,  1,  1], [  1,  0,  1], [  1,  1,  1] ]); 

    r,c=im.shape;

    h = np.double(h)

    cmtx = filter2(np.ones(im.shape), H/(dx*dx));

    A11 = im*(filter2(im, D2/(dx*dx), mode='nearest')-2*im/(dx*dx)) - h*cmtx; 
    A22 = im*(filter2(im, D2.transpose()/(dx*dx), mode='nearest')-2*im/(h*h)) - h*cmtx; 
    A12 = im*filter2(im, M/(dx*dx), mode='nearest'); 
    
    DetA = A11*A22-A12*A12;

    B11 = A22/DetA;
    B12 = -A12/DetA;
    B22 = A11/DetA;

    return B11, B12, B22

@jit('Tuple((float64[:,:],float64[:,:],float64))(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64,int64)',nopython=True,parallel=False)    
def helper(B11, B12, B22, bu, bv, u, v, r, c):
    unew = -(B11*bu+B12*bv);
    vnew = -(B12*bu+B22*bv);
    total_error = (np.linalg.norm(unew-u)+np.linalg.norm(vnew-v))/(r*c)    
    return unew, vnew, total_error
    
def physicsBasedOpticalFlowLiuShen(im1, im2, h, U, V):
    # new model
    #Dm=0*10**(-3);
    #f=Dm*laplacian(im1,1);
    f=0
    
    maxnum=60;
    tol = 1e-8;
    dx=1; 
    dt=1; # unit time

    # 
    # I: intensity function
    # Ix: partial derivative for x-axis
    # Iy: partial derivative for y-axis
    # It: partial derivative for time t
    # f: related all boundary assumption
    # lambda: regularization parameter
    # nb: the neighborhood information
    #
    #-------------------------------------------------------------------
    D  = np.array([[0, -1,  0], [0,  0,  0], [ 0, 1, 0] ])/2; # partial derivative 
    M  = np.array([[1,  0, -1], [0,  0,  0], [-1, 0, 1] ])/4; # mixed partial derivatives
    F  = np.array([[0,  1,  0], [0,  0,  0], [ 0, 1, 0] ]);   # average
    D2 = np.array([[0,  1,  0], [0, -2,  0], [ 0, 1, 0] ]);   # partial derivative
    H  = np.array([[1,  1,  1], [1,  0,  1], [ 1, 1, 1] ]); 
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    IIx = im1*filter2(im1, D/dx, mode='nearest');
    IIy = im1*filter2(im1, D.transpose()/dx, mode='nearest');
    II  = im1*im1;
    Ixt = im1*filter2((im2-im1)/dt, D/dx, mode='nearest');
    Iyt = im1*filter2((im2-im1)/dt, D.transpose()/dx, mode='nearest'); 
    
    k=0;
    total_error=100000000;
    u=np.float64(U);
    v=np.float64(V);

    r,c=im2.shape;

    #------------------------------------------------------------------
    B11, B12, B22 = generate_invmatrix(im1, h, dx);
    
    error=0;
    while total_error > tol and k < maxnum:
        bu = 2*IIx*filter2(u, D/dx, mode='nearest') + IIx*filter2(v, D.transpose()/dx, mode='nearest') + \
               IIy*filter2(v, D/dx, mode='nearest') + II*filter2(u, F/(dx*dx), mode='nearest') + \
               II*filter2(v, M/(dx*dx), mode='nearest') + h*filter2(u, H/(dx*dx))+Ixt;
    
        bv = IIy*filter2(u, D/dx, mode='nearest') + IIx*filter2(u, D.transpose()/dx, mode='nearest') + \
            2*IIy*filter2(v, D.transpose()/dx, mode='nearest') + II*filter2(u, M/(dx*dx), mode='nearest') + \
            II*filter2(v, F.transpose()/(dx*dx), mode='nearest') + h*filter2(v, H/(dx*dx))+Iyt;
        
        unew, vnew, total_error = helper(B11, B12, B22, bu, bv, u, v, r, c)
        #Do not use scipy.linalg.norm, use instead numpy.linalg.norm, because it is causing contentions and kernel yield,
        #scipy is making the kernel time much longer like 60% of the time, slowing down the machine, by severely increasing the
        #system load average.
        #total_error = (norm(unew-u,'fro')+norm(vnew-v,'fro'))/(r*c)
        
        u = unew;
        v = vnew;
        error=total_error;
        k=k+1  

    return np.float32(u), np.float32(v), error


