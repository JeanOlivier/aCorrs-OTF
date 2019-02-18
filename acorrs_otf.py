#!/bin/python
# -*- coding: utf-8 -*-

import ctypes
import os
import platform
from numpy.ctypeslib import ndpointer
from numpy import zeros, fromstring, ndarray, ceil, log2
from numpy import int8, uint8, int16, uint16
from ctypes import c_uint8, c_int8, c_uint16, c_int16, c_double, c_int, c_uint64

plat_info = dict(plat=platform.system())
if plat_info['plat'] == 'Windows':
    plat_info['lib'] = './acorrs.dll'
    plat_info['com'] = 'make acorrs.dll'
    # Adding cygwin libs path for windows
    libpath = r'C:\cygwin64\usr\x86_64-w64-mingw32\sys-root\mingw\bin\;'
    if libpath not in os.environ['PATH']:
        os.environ['PATH'] = libpath+os.environ['PATH']
    
else:
    plat_info['lib'] = './acorrs.so'
    plat_info['com'] = 'make acorrs.so'


if not os.path.isfile(plat_info['lib']):
    raise IOError("{lib} is missing. To compile on {plat}:\n{com}\n".format(**plat_info))

lib = ctypes.cdll[plat_info['lib']]

def set_mpreal_precision(digits):
    lib.set_mpreal_precision(digits)

# Applies to instances created afterwards
set_mpreal_precision(100)

# OpenMP stuff
if plat_info["plat"] == "Windows":
    omp = ctypes.CDLL('libgomp-1')
else:
    try:
        omp = ctypes.CDLL("/usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so")
    except:
        omp = ctypes.CDLL("libgomp.so")
set_num_threads = omp.omp_set_num_threads
set_num_threads.argtypes=(ctypes.c_int,)
set_num_threads.restype=None
get_num_threads = omp.omp_get_max_threads
get_num_threads.restype=ctypes.c_int
get_num_threads.argtypes=None

# Autocorrelations
class ACorrUpTo():
    def __init__(self, k, data="uint8", fft=None, fftchunk=8192, k_fft=105):
        if type(data) is ndarray:
            self.dtype = data.dtype.name
        else:
            self.dtype = data

        if fft is None: 
            if k>k_fft:  # k_fft is empirical for each system 
                fft = True
            else:
                fft = False

        fftflag = "FFT" if fft else ""
        if fft and k>fftchunk:
            fftchunk = int(2**ceil(log2(k))) # Ceil to power of two
        
        self.atype = dict(uint8 = c_uint8, int8 = c_int8, uint16 = c_uint16, int16 = c_int16)
        self.k = k
        self.res = zeros(self.k)
        self.len = fftchunk
        
        assert self.dtype in self.atype.keys(), TypeError("Unsupported type ({})".format(self.dtype))
        
        # Accessing the proper C functions
        self._init = lib["ACorrUpTo{fft}_{dtype}_init".format(dtype=self.dtype, fft=fftflag)]
        self.accum = lib["ACorrUpTo{fft}_{dtype}_accumulate".format(dtype=self.dtype, fft=fftflag)]
        self.aCorrs = lib["ACorrUpTo{fft}_{dtype}_get_aCorrs".format(dtype=self.dtype, fft=fftflag)]
        self.dest = lib["ACorrUpTo{fft}_{dtype}_destroy".format(dtype=self.dtype, fft=fftflag)]
        self._init.argtypes = (c_int, c_int)
        #self._init.argtypes = (c_int,)
        self.aCorrs.argtypes = (ndpointer(c_double, shape=(k,)), c_int)

        self.accumulate_rk = lib["ACorrUpTo{fft}_{dtype}_accumulate_rk".format(dtype=self.dtype, fft=fftflag)]
        self.accumulate_rk.argtypes = (
            ndpointer(dtype=self.atype[self.dtype], shape=(len(data),)),
            c_uint64,
            ndpointer(c_double, shape=(k,)), 
            c_int
        )
       
        #self.accumulate_rk_FFTW = lib["ACorrUpTo_{dtype}_accumulate_rk_FFTW".format(dtype=self.dtype)]
        #self.accumulate_rk_FFTW.argtypes = (
        #    ndpointer(dtype=self.atype[self.dtype], shape=(len(x),)),
        #    c_uint64,
        #    ndpointer(c_double, shape=(k,)), 
        #    c_int
        #)

        # Actually initializing the C++ class
        self.init()
        
        # If data is an array we accumulate it
        if type(data) is ndarray:
            self(data)
            

    def init(self):
        self._init(self.k, self.len)
        #self._init(self.k)

    def accumulate(self, x):
        # Accumulating
        self.accum.argtypes = (ndpointer(dtype=self.atype[self.dtype], shape=(len(x),)), c_uint64)
        self.accum(x, len(x))
        # Updating internal result variable (self.res)
        self.compute_aCorrs() # ~30µs, worth updating self.res often

    def compute_aCorrs(self):
        self.aCorrs(self.res, self.k)
        
    def get_aCorrs(self):
        self.compute_aCorrs()
        return self.res

    def destroy(self):
        self.dest()

    def __call__(self, x):
        self.accumulate(x)

    # TODO: Actual representation, with length and type maybe?
    #def __repr__(self):
    #    return self.res.__repr__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        self.destroy()

    def __del__(self):
        self.destroy()


# Python functions for testing:
def rk_py(x,k):
    n = len(x)
    c = zeros(k, double)
    for i in range(n-k):
            for j in range(k):
                    c[j] += double(x[i])*x[i+j]
    return double(c)

def rk(x,k):
    res = zeros(k)
    with ACorrUpTo(k,x.dtype.name) as aa:
        aa.accumulate_rk(x, len(x)-k, res, k)
    return res

#def rk_FFTW(x,k):
#    res = zeros(k)
#    with ACorrUpTo(k,x.dtype.name) as aa:
#        aa.accumulate_rk_FFTW(x, len(x)-k, res, k)
#    return res



def rkFFT(x,lags):
        '''fft, pad 0s, non partial'''
        n=len(x)
        # pad 0s to 2n-1
        ext_size=2*n-1
        # nearest power of 2
        fsize=2**numpy.ceil(numpy.log2(ext_size)).astype('int')
        # do fft and ifft
        cf=numpy.fft.rfft(x,fsize)
        sf=cf.conjugate()*cf
        corr=numpy.fft.irfft(sf)
        corr=corr[:lags]
        return corr

def rkFFT_long(x,k,l=8192,parallel=8):
    #TODO: Implement in C++ with FFTW
    # Python parallel version only faster if k is long.
    # It should be N*log(l) complexity vs N*k for direct rk.
    if k>l/2:
        l=2*k+1
    n = len(x)
    num = n//l
    kfactors = array([double(n)/(n-j*num) for j in range(k)])
    y = x[:x.size-x.size%l]
    y.resize(num,l)
    if parallel:
        from pathos.multiprocessing import Pool
        p = Pool(parallel)
        res = sum(p.map(lambda i:rkFFT(i,k),y), axis=0)
        del p
    else:
        res = zeros(k)
        for i in y:
            res += rkFFT(i,k)
    return res*kfactors


def aCorr(x,k):
    n = len(x)
    c = zeros(k)
    for i in range(n-k):
            for j in range(k):
                    c[j] += ((x[i]-x.mean())*(x[i+j]-x.mean()))/(len(x)-j)
    for i in range(n-k, n):
            for j in range(n-i):
                    c[j] += ((x[i]-x.mean())*(x[i+j]-x.mean()))/(len(x)-j)
    return c

def aCorrFFT(x,k):
    b = rfft(x)
    b[0]=0 # This removes the mean
    c = b*b.conj()
    d = irfft(c)
    return array([i*x.size/(x.size-j) for j,i in enumerate(d)]) # Correct-k normalization

def autocorr3(x,lags):
        '''fft, pad 0s, non partial'''
        n=len(x)
        # pad 0s to 2n-1
        ext_size=2*n-1
        # nearest power of 2
        fsize=2**numpy.ceil(numpy.log2(ext_size)).astype('int')
        xp=x-numpy.mean(x)
        var=numpy.var(x)
        # do fft and ifft
        cf=numpy.fft.rfft(xp,fsize)
        sf=cf.conjugate()*cf
        corr=numpy.fft.irfft(sf)
        corr=corr[:lags]
        return array([i/(n-j) for j,i in enumerate(corr)])

