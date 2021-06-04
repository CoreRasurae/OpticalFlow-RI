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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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
import struct
from decimal import *

softOne = Decimal(struct.unpack('d', struct.pack('l', 1023 << 52))[0])
softZero = Decimal(struct.unpack('d', struct.pack('l', 0))[0])

def simpleGaussianKernel(n, sigma):
    print('Sigma', sigma)
    values = np.zeros([n], dtype=np.float64)
    n2 = np.int((n - 1) / 2)
    x = 1 - n
    for i in np.arange(0, n2):
    	values[i] = np.exp(- x * x * 0.125 / (sigma * sigma))
    	values[n - i - 1] = values[i]
    	x += 2
    #Now we have the values for the left side and right sides, which are symmetrical
    if n & 1 == 1:
         #Lets place the center value which is 1 = e^(- 0 / (8*sigma^2))
         values[n2] = 1.0
    #Now lets normalize
    values /= sum(values)
    return sum(values), values
    
def simpleGaussianKernelDecimal(n, sigma):
    if sigma <= 0:
        if n == 1:
            return softOne, np.array([softOne])
        elif n == 3:
            result = np.array( [ struct.unpack('d', struct.pack('l', 0x3fd0000000000000))[0], #0.25
                                 struct.unpack('d', struct.pack('l', 0x3fe0000000000000))[0], #0.50
                                 struct.unpack('d', struct.pack('l', 0x3fd0000000000000))[0], #0.25
                               ], dtype=np.float64 )
            return softOne, result
        elif n == 5:
            result = np.array( [ struct.unpack('d', struct.pack('l', 0x3fb0000000000000))[0], #0.0625
                                 struct.unpack('d', struct.pack('l', 0x3fd0000000000000))[0], #0.25
                                 struct.unpack('d', struct.pack('l', 0x3fd8000000000000))[0], #0.375
                                 struct.unpack('d', struct.pack('l', 0x3fd0000000000000))[0], #0.25
                                 struct.unpack('d', struct.pack('l', 0x3fb0000000000000))[0], #0.0625
                               ], dtype=np.float64 )
            return softOne, result
        elif n == 7:
            result = np.array( [ struct.unpack('d', struct.pack('l', 0x3fa0000000000000))[0], #0.03125
                                 struct.unpack('d', struct.pack('l', 0x3fbc000000000000))[0], #0.109375
                                 struct.unpack('d', struct.pack('l', 0x3fcc000000000000))[0], #0.21875
                                 struct.unpack('d', struct.pack('l', 0x3fd2000000000000))[0], #0.28125
                                 struct.unpack('d', struct.pack('l', 0x3fcc000000000000))[0], #0.21875
                                 struct.unpack('d', struct.pack('l', 0x3fbc000000000000))[0], #0.109375
                                 struct.unpack('d', struct.pack('l', 0x3fa0000000000000))[0], #0.03125
                               ], dtype=np.float64 )
            return softOne, result
        elif n == 9:
            result = np.array( [ struct.unpack('d', struct.pack('l', 0x3f90000000000000))[0], #4  / 256
                                 struct.unpack('d', struct.pack('l', 0x3faa000000000000))[0], #13 / 256
                                 struct.unpack('d', struct.pack('l', 0x3fbe000000000000))[0], #30 / 256
                                 struct.unpack('d', struct.pack('l', 0x3fc9800000000000))[0], #51 / 256
                                 struct.unpack('d', struct.pack('l', 0x3fce000000000000))[0], #60 / 256
                                 struct.unpack('d', struct.pack('l', 0x3fc9800000000000))[0], #51 / 256
                                 struct.unpack('d', struct.pack('l', 0x3fbe000000000000))[0], #30 / 256
                                 struct.unpack('d', struct.pack('l', 0x3faa000000000000))[0], #13 / 256
                                 struct.unpack('d', struct.pack('l', 0x3f90000000000000))[0], #4  / 256
                              ], dtype=np.float64 )
            return softOne, result     

    values = np.zeros([n], dtype=Decimal)
    n2 = np.int((n - 1) / 2)
    x = 1 - n
    for i in np.arange(0, n2):
    	values[i] = Decimal(np.exp(- Decimal(x * x) * Decimal(0.125) / Decimal(sigma * sigma)))
    	values[n - i - 1] = values[i]
    	x += 2
    #Now we have the values for the left side and right sides, which are symmetrical
    if n & 1 == 1:
         #Lets place the center value which is 1 = e^(- 0 / (8*sigma^2))
         values[n2] = Decimal(1.0)
    #Now lets normalize
    values /= sum(values)
    return sum(values), values
