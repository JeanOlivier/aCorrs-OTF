#ifndef autocoFFT_H
#define autocoFFT_H

#include "common.hpp"
#include "acorrs.hpp"
#include "fftw3.h"

inline void halfcomplex_norm2(double *buff, int fftwlen);

template <class T> class ACorrUpToFFT: public ACorrUpTo<T> 
{
public:
// Base class stuff //
    typedef typename ACorrUpTo<T>::accumul_t accumul_t;
    accumul_t *rk = ACorrUpTo<T>::rk;
    accumul_t &m = ACorrUpTo<T>::m; // Some voodoo here; needed for openmp reduction?
    int &k = ACorrUpTo<T>::k;
    mpreal *rk_mpfr = ACorrUpTo<T>::rk_mpfr;
    // FFT(W) specific stuff
    int len;
    int fftwlen;
    fftw_plan fwd_plan;
    fftw_plan rev_plan;
    double *in;
    double *out;
    int counter_max;
// Constructors //
    ACorrUpToFFT(int k, int len);

    void accumulate_m_rk(T*, uint64_t);
    int compute_accumul_max();
    int test;

    virtual ~ACorrUpToFFT();
};


#endif // autocoFFT_H
