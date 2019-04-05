#!/bin/python
# -*- coding: utf-8 -*-

import sys, os, platform
from numpy import ndarray

# Setting up the proper libraries and paths, mainly for Windows support
libpath = os.path.abspath(os.path.dirname(__file__))
plat_info = dict(plat=platform.system())
if plat_info['plat'] == 'Windows':
    plat_info['lib'] = os.path.join(libpath, 'acorrs_wrapper.pyd')
    plat_info['com'] = 'make acorrs_wrapper.pyd'
    # Adding cygwin libs path for windows
    libspath = 'C:\\cygwin64\\usr\\x86_64-w64-mingw32\\sys-root\\mingw\\bin'
    if libspath not in os.environ['PATH']:
        os.environ['PATH'] = libspath+os.path.pathsep+os.environ['PATH']   
else:
    plat_info['lib'] = os.path.join(libpath, 'acorrs_wrapper.so')
    plat_info['com'] = 'make acorrs_wrapper.so'

if not os.path.isfile(plat_info['lib']):
    raise IOError("{lib} is missing. To compile on {plat}:\n{com}\n".format(**plat_info))

import acorrs_wrapper

# Applies to instances created afterwards
acorrs_wrapper.set_mpreal_precision(100)

# Returns the proper class. Fancy name: factory. Ghetto name: wrapper wrapper.
def ACorrUpTo(k, data, fft=None, fftchunk=8192, k_fft=105):
    if type(data) is ndarray:
        dtype = data.dtype.name
    else:
        dtype = data

    if fft is None: 
        if k>k_fft:  # k_fft is empirical for each system 
            fft = True
        else:
            fft = False

    if fft and k>fftchunk:
        fftchunk = int(2**ceil(log2(k))) # Ceil to power of two
    
    classname = "ACorrUpTo{fft}_{dtype}".format(dtype=dtype, fft="FFT" if fft else "")
    
    if fft:
        retClass = getattr(acorrs_wrapper, classname)(k, fftchunk)
    else:
        retClass = getattr(acorrs_wrapper, classname)(k)
    
    if type(data) is ndarray:
        retClass(data)
    
    return retClass

