#!/bin/python
# -*- coding: utf-8 -*-

import sys, os, platform, time
import numpy as np
from numpy import uint8, int8, uint16, int16, double
from numpy import ndarray, ceil, log2, iinfo, zeros, allclose, arange, array 
from numpy import floor, log10, savez_compressed, load
from matplotlib import pyplot as plt

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
def ACorrUpTo(k, data, fft=None, fftchunk=8192, k_fft=44):
    if type(data) is ndarray:
        dtype = data.dtype.name
    else:
        dtype = data

    if fft is None: 
        if k>=k_fft:  # k_fft is empirical for each system 
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


# Testing utilities!
def timefunc(fct, *args, **kwargs):
    start = time.time()
    res = fct(*args, **kwargs)
    stop = time.time()
    return stop-start, res


def get_rand_array(dtype,length):
    i = iinfo(dtype)
    return np.random.randint(low=i.min, high=i.max, size=(1, length), dtype=dtype)


def _get_scaling(k, dtype, size, verbose=True, verb_prefix='', data_func=get_rand_array):
    if np.isscalar(k):
        ks = arange(k)+1
    else:
        ks = array(k)
    kmax = ks.max()
    
    r = data_func(dtype, size)
    res = zeros((kmax, kmax), double)
    func = lambda k, x, fft: ACorrUpTo(k, x, fft=fft).res


    timesd = []
    timesf = []
    errlmean = []
    errlmax = []
    for i,k in enumerate(ks):
        if verbose:
            print '{p:}Iteration: {i:0{f:d}d}/{l:d}'.format(f=int(floor(log10(len(ks))))+1, i=i+1, l=len(ks), p=verb_prefix)
        td,ad = timefunc(func, k, r, fft=False)
        tf,af = timefunc(func, k, r, fft=True)

        assert allclose(ad,af), "Direct/FFT results aren't close!"
        timesd += [td]
        timesf += [tf]
        errlmean += [(abs((af-ad)/ad)).mean()]
        errlmax += [(abs((af-ad)/ad)).max()]


    return array([timesd, timesf, errlmean, errlmax, ks])


def get_scaling(k, dtype, size, n=10, verbose=True, *args, **kwargs):
    if np.isscalar(k):
        ks = arange(k)+1
    else:
        ks = array(k)
    kmax = ks.max()

    res = zeros((n,5,len(ks)))

    for i in range(n):
        prfx = ''
        if verbose:
            prfx = 'Iteration: {i:0{f:d}d}/{l:d}  |  Sub-'.format(f=int(floor(log10(n)))+1,i=i+1,l=n)
        res[i] = _get_scaling(k, dtype, size, verbose=verbose, verb_prefix=prfx, *args, **kwargs)

    return res, dtype, size

def get_gp_order(dtype, size):
    if size >= 2**30:
        gp = 'G'
        order = 2**30
    elif size >= 2**20:
        gp = 'M'
        order = 2**20
    elif size >= 2**10:
        gp = 'k'
        order = 2**10
    return gp,order


def save_scaling(*args, **kwargs):
    dtype = args[1]
    size = args[2]
    gp,order = get_gp_order(dtype, size)
    fn = 'Scaling_{:s}_{:.5g}{:s}iSa.npz'.format(np.dtype(dtype).name, size/order, gp)
    tmp = get_scaling(*args, **kwargs)
    savez_compressed(fn, tmp)
    return tmp

def load_scaling(fn):
    return load(fn)['arr_0']

# E.g.
# >>> a=save_scaling('S',1024,int16,16*2**30,n=20)
# >>> plot_scaling(*a, save=True)
# >>> plot_errors(*a, save=True)
def plot_scaling(res, dtype, size, save=False, comment=''):
    rmean = res.mean(axis=0)
    rstd = res.std(axis=0)
    rmax = res.max(axis=0)
    gp, order = get_gp_order(dtype, size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,r in enumerate(res):
        ax.plot(r[-1], r[0], '-', color='C0', alpha=0.1)
        ax.plot(r[-1], r[1], '-', color='C1', alpha=0.1)
    ax.plot(rmean[-1], rmean[0], '-', label='Direct Product')
    ax.plot(rmean[-1], rmean[1], '-', label='FFT Convolution')

    ax.legend()
    ax.grid(True)

    ax.set_xlabel('Number of autocorrelations (k)')
    ax.set_ylabel('Computation time (s)')
   
    ax.set_title('Avg. {:d} | {:.5g} {:s}iSa | dtype {:s} {:s}'.format(res.shape[0], size/order, gp, np.dtype(dtype).name, comment))

    if save:
        fig.savefig('Scaling_{}_{:.5g}{:s}iSa.pdf'.format(np.dtype(dtype).name, size/order, gp), bbox_inches='tight')


def plot_errors(res, dtype, size, save=False, comment=''):
    rmean = res.mean(axis=0)
    rstd = res.std(axis=0)
    rmax = res.max(axis=0)
    gp, order = get_gp_order(dtype, size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,r in enumerate(res):
        ax.plot(r[-1], abs(r[2]), '-', color='C0', alpha=0.1)
        ax.plot(r[-1], abs(r[3]), '-', color='C1', alpha=0.1)
    ax.plot(rmean[-1], abs(rmean[2]), '-', label='Average')
    ax.plot(rmean[-1], abs(rmax[3]), '-', label='Worse')

    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')

    ax.set_xlabel('Number of autocorrelations (k)')
    ax.set_ylabel('FFT Convolution Relative Error')
   
    ax.set_title('Avg. {:d} | {:.5g} {:s}iSa | dtype {:s} {:s}'.format(res.shape[0], size/order, gp, np.dtype(dtype).name, comment))

    if save:
        fig.savefig('Errors_{}_{:.5g}{:s}iSa.pdf'.format(np.dtype(dtype).name, size/order, gp), bbox_inches='tight')






    

