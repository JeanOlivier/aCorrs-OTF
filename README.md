# aCorrs-OTF
C++ library with python interface to compute autocorrelations on the fly. Targets many-cores workstation for near-realtime processing of very large data acquisition. 


## Description
`common.hpp/cpp` contains general includes and functions to set mpreal precision and handle lots of threads on Windows.
`acorrs.hpp/cpp` provide class templates to compute the autocorrelation of a signal for the *k* first lags/delays using direct products.
`acorrsFFT.hpp/cpp` is an optimised version of `acorrs.hpp/cpp` using FFT convolutions.
`acorrs._wrapper.cpp` is a pybind11 binding of `acorrs/acorrsFFT`.
`acorrs_otf.py` provides the `ACorrUpTo` factory for convenience of use.


## Features:
- Supports Windows 10 and Linux.
- Supports 8 and 16 bit data both signed and unsigned
- Uses [MPFR C++](http://www.holoborodko.com/pavel/mpfr/) for precision when handling huge numbers
- Uses OpenMP to take advantage of workstation hardware
  - Handles thread affinity on Windows for system with over 64 logical cores
- Uses direct multiplications for small number of autocorrelations and convolutions with FFTW for large numbers.
  - Same results regardless of algorithm (corrects for the inter-fft_buffers correlations up to machine precision)
- Can be fed blocks of data successively and compute the results afterwards (correlations across blocks are ignored)
- Refactored for better structure and compilation times.


## Algorithm and optimisation
The algorithm used is the following:

![](https://latex.codecogs.com/gif.latex?a_k=\frac{1}{N-k}\sum_{i=1}^{N-k}(x_i-\mu_0)(x_{i+k}-\mu_k)\quad\quad\text{with}\quad\quad\mu_j=\frac{1}{N-k}\sum_{i=1+j}^{N-k+j}x_i)

where ![](https://latex.codecogs.com/gif.latex?a_k) is the autocorrelation with lag ![](https://latex.codecogs.com/gif.latex?k), ![](https://latex.codecogs.com/gif.latex?x_i) is the ![](https://latex.codecogs.com/gif.latex?i^\text{th}) point of the signal, ![](https://latex.codecogs.com/gif.latex?N) is the signal length, and ![](https://latex.codecogs.com/gif.latex?\mu_j) is its partial mean computed on ![](https://latex.codecogs.com/gif.latex?x_{1+j}) through ![](https://latex.codecogs.com/gif.latex?x_{N-k+j}) with condition ![](https://latex.codecogs.com/gif.latex?0\leq%20j\leq%20k).

We re-express it as :

![](https://latex.codecogs.com/gif.latex?a_k=\frac{r_k}{N-k}-\left(\frac{M-\beta_k}{N-k}\right)\cdot\left(\frac{M-\gamma_k}{N-k}\right)),

with 

![](https://latex.codecogs.com/gif.latex?r_k%20%3D%20%5Csum%5E%7BN-k%7D_%7Bi%3D1%7D%20x_ix_%7Bi&plus;k%7D%20%5Cquad%3B%5Cquad%20%5Cbeta_k%20%3D%20%5Csum_%7Bi%3DN-k&plus;1%7D%5E%7BN%7D%20x_i%20%5Cquad%3B%5Cquad%20%5Cgamma_k%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bk%7D%20x_i%20%5Cquad%5Ctext%7Band%7D%5Cquad%20M%3D%5Csum_%7Bi%3D1%7D%5EN%20x_i),

in order to speed up calculations with simple parallelizable forms for ![](https://latex.codecogs.com/gif.latex?r_k) and ![](https://latex.codecogs.com/gif.latex?M) and correcting for the induced errors using the relatively simple ![](https://latex.codecogs.com/gif.latex?\gamma_k) and ![](https://latex.codecogs.com/gif.latex?\beta_k).

In the ![](https://latex.codecogs.com/gif.latex?k=0) case, it falls back on the variance and the corrections vanish.

## Example:
```python
from numpy import fromfile, uint16
from aCorrs_OTF import ACorrUpTo

x = fromfile("/path/to/potentially/huge/file", uint16)
a = ACorrUpTo(500, x, fft=True, fftchunk=8192) # Initialising and accumulating data

print a.res
```
Output:
```python
[  3.63332284e+08   7.98651587e+06   7.81700302e+06   7.61148771e+06
   7.56474479e+06   7.58548592e+06   7.60689215e+06   7.66726265e+06
                                  ...
   6.30917894e+06   6.24041607e+06   6.24923766e+06   6.25858389e+06
   6.27447189e+06   6.23658121e+06   6.22335777e+06   6.24173146e+06]

```

## Compiling

Simply execute `make all`.
