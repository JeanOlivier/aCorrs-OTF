#ifndef autoco_H
#define autoco_H

#if defined(__CYGWIN__) || defined(__MINGW64__)
    // see number from: sdkddkver.h
    // https://docs.microsoft.com/fr-fr/windows/desktop/WinProg/using-the-windows-headers
    #define _WIN32_WINNT 0x0602 // Windows 8
    #include <windows.h>
    #include <Processtopologyapi.h>
    #include <processthreadsapi.h>
#endif

#define NOOP (void)0

#include <stdio.h> 
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"

#include <iostream>
#include <iomanip>

#include <omp.h>
#include <ctime>
#include <limits>


#ifdef _WIN32_WINNT
    #include "mpreal.h"
#else
    #include <mpreal.h>
#endif

//#include "gmp.h"
//#include "mpfr.h"


// Timing for benchmarking
//#include <time.h>


#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace std;
using mpfr::mpreal;

// For setting desired mpreal precision beforehand
void set_mpreal_precision(int d){
    // Before any mreal are created
    const int digits = d; // Setting high precision
    mpreal::set_default_prec(mpfr::digits2bits(digits));
}

void manage_thread_affinity()
{
    #ifdef _WIN32_WINNT
        int nbgroups = GetActiveProcessorGroupCount();
        int *threads_per_groups = (int *) malloc(nbgroups*sizeof(int));
        for (int i=0; i<nbgroups; i++)
        {
            threads_per_groups[i] = GetActiveProcessorCount(i);
        }

        // Fetching thread number and assigning it to cores
        int tid = omp_get_thread_num(); // Internal omp thread number (0 -- OMP_NUM_THREADS)
        HANDLE thandle = GetCurrentThread();
        bool result;
        
        WORD set_group = tid%nbgroups; // We change group for each thread
        int nbthreads = threads_per_groups[set_group]; // Nb of threads in group for affinity mask.
        GROUP_AFFINITY group = {((uint64_t)1<<nbthreads)-1, set_group}; // nbcores amount of 1 in binary
        
        result = SetThreadGroupAffinity(thandle, &group, NULL); // Actually setting the affinity
        if(!result) fprintf(stderr, "Failed setting output for tid=%i\n", tid);
    #else
        //We let openmp and the OS manage the threads themselves
    #endif
}


//TODO: Make (it?) a general correlation class (with aCorr as a special case?)
template<class T>
class ACorrUpTo{
public:
// Variables //
    
    // Casting over different type sign is very slow, we avoid it.
    // (T)-1>0 is true only if T is unsigned
    // Would work with int < 64bits but would overflow faster
    typedef typename conditional<((T)-1>0), uint64_t, int64_t>::type accumul_t;
    
    // Math stuff 
    accumul_t m;
    accumul_t n;
    int k;

    accumul_t *rk;
    accumul_t *bk;
    accumul_t *gk;
    
    // Precision stuff
    mpreal m_mpfr;
    mpreal n_mpfr;
    mpreal k_mpfr;

    mpreal *rk_mpfr;
    mpreal *bk_mpfr;
    mpreal *gk_mpfr;

    // Autocorrelations results
    mpreal *aCorrs_mpfr;
    double *aCorrs;

    // Managerial stuff
    uint64_t chunk_processed;
    uint64_t chunk_size;
    uint64_t block_processed;

// Constructors //
    ACorrUpTo(int k):
        m(0), n(0), k(k), m_mpfr(0), n_mpfr(0), k_mpfr(k)
    {
        rk = new accumul_t [k](); // Parentheses initialize to zero
        gk = new accumul_t [k]();
        bk = new accumul_t [k]();
        rk_mpfr = new mpreal [k]();   
        gk_mpfr = new mpreal [k]();
        bk_mpfr = new mpreal [k]();   

        aCorrs = new double [k]();
        aCorrs_mpfr = new mpreal [k]();   
        
        block_processed = 0;
        chunk_processed = 0;
        chunk_size = compute_chunk_size(); // Auto largest possible
    }

// Methods //
    void accumulate(T *buffer, uint64_t size){
        // On each call, a new block being processed.
        block_processed++;
        n += size;
        accumulate_gk(buffer, size); // Compute bk on very first data
        // Loop on whole chunks
        uint64_t i; // Will point to last partial chunk after the loop 
        for (i=0; i<(size-k)/chunk_size; i++){
            accumulate_chunk(buffer+i*chunk_size, chunk_size);
            update(); // Update mpfr values and reset accumulators
        }
        // Last (potentially) partial chunk, keeps chunk_processed accurate
        if ((size-k)%chunk_size){
            accumulate_chunk(buffer+i*chunk_size, (size-k)%chunk_size);
            update(); // Update mpfr values and reset accumulators
        }
        // Right edge chunk, doesn't count in chunk_processed because it's small
        accumulate_chunk_edge(buffer, size);
        update(); // Update mpfr values and reset accumulators
        accumulate_bk(buffer, size); // Computing (replacing) bk on last data
    }

    inline void accumulate_chunk(T *buffer, uint64_t size){
        accumulate_m_rk(buffer, size); // Accumulating
        chunk_processed++;
    }

    inline void accumulate_chunk_edge(T *buffer, uint64_t size){
        accumulate_m_rk_edge(buffer, size); // Accumulating
        //chunk_processed++;
    }

    //inline void accumulate_m_rk(T *buffer, uint64_t size);
    virtual inline void accumulate_m_rk(T *buffer, uint64_t size){
        #pragma omp parallel
        {
            manage_thread_affinity();
            #pragma omp for simd reduction(+:m), reduction(+:rk[:k])
            for (uint64_t i=0; i<size; i++){
                m += (accumul_t)buffer[i];
                #pragma omp ordered simd
                for (int j=0; j<k; j++){
                    rk[j] += (accumul_t)buffer[i]*(accumul_t)buffer[i+j];
                }
            }
        }
    }

    inline void accumulate_m_rk_edge(T *buffer, uint64_t size){
        for (uint64_t i=size-k; i<size; i++){
            m += (accumul_t)buffer[i];
            for (uint64_t j=0; j<size-i; j++){
                rk[j] += (accumul_t)buffer[i]*(accumul_t)buffer[i+j];
            }
        }
    }

    // gk is the beginning corrections 
    // It should be computed ONCE on the very first chunk of data
    inline void accumulate_gk(T *buffer, uint64_t size){
        for (int i=0; i<k; i++){
            for (int j=0; j<i; j++){
                gk[i] += (accumul_t)buffer[j];
            }
            gk_mpfr[i] += gk[i]; // Updating precision value
            gk[i] = 0; // Reseting accumulator
        }
    }

    // bk is the end corrections
    // Re-compute for each new chunk to keep autocorr valid 
    inline void accumulate_bk(T *buffer, uint64_t size){
        for (int i=0; i<k; i++){
            for (uint64_t j=size-i; j<size; j++){
                bk[i] += (accumul_t)buffer[j];
            }
            bk_mpfr[i] += bk[i]; // Updating precision value
            bk[i] = 0; // Reseting accumulator
        }
    }
    
    inline void update(){
        update_mpfr();
        reset_accumulators();
    }

    inline void update_mpfr(){
        m_mpfr += m;        
        n_mpfr += n;
        for (int i=0; i<k; i++){
            rk_mpfr[i] += rk[i]; 
        } // bk_mpfr and gk_mpfr are updated at computation
    }

    inline void reset_accumulators(){
        m=0;
        n=0;
        for (int i=0; i<k; i++){
            rk[i] = 0;
        }
    }

    mpreal get_mean_mpfr(){
        update(); // Just to be sure
        mpreal r = m_mpfr/n_mpfr;
        return r;
    }

    double get_mean(){
        return (double)get_mean_mpfr();
    }
    
    mpreal get_var_mpfr(){
        update(); // Just to be sure
        mpreal v = (rk_mpfr[0]-pow(m_mpfr,2)/n_mpfr)/(n_mpfr);
        return v;
    }

    double get_var(){
        return (double)get_var_mpfr();
    }

    mpreal* get_aCorrs_mpfr(){
        // No corr across blocks: i -> i*block_processed
        mpreal n_k;
        for (int i=0; i<k; i++){
            n_k = n_mpfr - (mpreal)(i*block_processed);
            aCorrs_mpfr[i] = (rk_mpfr[i] - (m_mpfr-bk_mpfr[i])*(m_mpfr-gk_mpfr[i])/n_k)/n_k;
        }
        return aCorrs_mpfr; // Return pointer to array
    }


    void compute_aCorrs(){
        get_aCorrs_mpfr();
        for (int i=0; i<k; i++){
            aCorrs[i] = (double)aCorrs_mpfr[i]; // Small loss of precision here
        }
    }
    
    double* get_aCorrs(){
        compute_aCorrs();
        return aCorrs; // Return pointer to array
    }

    void get_aCorrs(double* res, int size){
        compute_aCorrs();
        for (int i=0; i<size; i++){
            res[i] = aCorrs[i];
        }
    }

    void get_rk(double* res, int size){
        if (size>k){
            size = k;
        }
        for (int i=0; i<size; i++){
            res[i] = (double)rk[i];
        }
    }

   
    // Max chunk_size to avoid overflow: chunk_size*buff_max² == accumul_max
    uint64_t compute_chunk_size(){
        uint64_t buff_max = numeric_limits<T>::max();
        uint64_t accumul_max = numeric_limits<accumul_t>::max();
        uint64_t ret = accumul_max/(buff_max*buff_max);
        return ret; // Int division removes fractional possibility
    }


// Destructor //
   virtual ~ACorrUpTo(){
       delete[] rk;
       delete[] gk;
       delete[] bk;
       delete[] rk_mpfr;
       delete[] gk_mpfr;
       delete[] bk_mpfr;
   }

};


inline void halfcomplex_norm2(double *buff, int fftwlen){
    // Multiplying with conjugate in-place
    buff[0] = buff[0]*buff[0]; // By symmetry, first one is purely real.
    int j=0;
    // buff*buff.conj() (result is real), buff in half-complex format
    for (j++; j<fftwlen/2; j++){
        // (a+ib)*(a-ib) = a²+b²
        buff[j] = buff[j]*buff[j] + buff[fftwlen-j]*buff[fftwlen-j]; // norm²
    }
    buff[j] = buff[j]*buff[j]; // fftwlen even implies n/2 is purely real too
    // buff*buff.conj() (imaj part of result)
    for (j++; j<fftwlen; j++){
        buff[j] = 0;   // Norm² of complex has no imaj part
    }
}


template <class T>
class ACorrUpToFFT: public ACorrUpTo<T> 
{
public:
    // Base class stuff
    typedef typename ACorrUpTo<T>::accumul_t accumul_t;
    accumul_t *rk = ACorrUpTo<T>::rk;
    accumul_t &m = ACorrUpTo<T>::m;
    int &k = ACorrUpTo<T>::k;
    mpreal *rk_mpfr = ACorrUpTo<T>::rk_mpfr;
    // FFT(W) specific stuff
    int len;
    int fftwlen;
    fftw_plan fwd_plan;
    fftw_plan rev_plan;
    double *in;
    double *out;

    ACorrUpToFFT(int k, int len): ACorrUpTo<T>(k), len(len)
    {
        // FFT length 
        fftwlen = 1<<(int)ceil(log2(2*len-1)); //TODO: Assert that k < len
        
        // Tries to load wisdom if it exists
        fftw_import_wisdom_from_filename("FFTW_Wisdom.dat");
        
        // Generating FFTW plans
        in = fftw_alloc_real(fftwlen); // Temp buffers for plan
        out = fftw_alloc_real(fftwlen);
        fwd_plan = fftw_plan_r2r_1d(fftwlen, in, out, FFTW_R2HC, FFTW_EXHAUSTIVE);
        rev_plan = fftw_plan_r2r_1d(fftwlen, in, out, FFTW_HC2R, FFTW_EXHAUSTIVE);
    }

    inline void accumulate_m_rk(T*, uint64_t);

    virtual ~ACorrUpToFFT(){
        // Saving wisdom for future use
        fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
        // Deleting plans
        fftw_destroy_plan(fwd_plan);
        fftw_destroy_plan(rev_plan);
        // Deleting temp buffers
        fftw_free(in); 
        fftw_free(out);
    }
};


template <class T> 
inline void ACorrUpToFFT<T>::accumulate_m_rk(T *buffer, uint64_t size){
    uint64_t fftnum = size/len;
    #pragma omp parallel
    {
        manage_thread_affinity();
        double *ibuff = fftw_alloc_real(fftwlen);
        double *obuff = fftw_alloc_real(fftwlen);
        double *rk_fft_local = fftw_alloc_real(fftwlen);
        
        #pragma omp for reduction(+:m), reduction(+:rk[:k])
        for (uint64_t i=0; i<fftnum; i++){
            T *buff = buffer + i*len;
            // Filling buffers and accumulating m
            int j;
            for (j=0; j<len; j++){
                m += (accumul_t)buff[j];
                ibuff[j] = (double)buff[j];
            }
            for(; j<fftwlen; j++){
                ibuff[j] = 0; // Needs zeroing, used as scratch pad
            }
        
            fftw_execute_r2r(fwd_plan, ibuff, obuff); // Forward FFT
            halfcomplex_norm2(obuff, fftwlen); // obuff*obuff.conj() element-wise
            // Reverse FFT used to be here instead of outside this loop
    
            // Accumulating rk, correcting for the missing data between fft_chunks
            for (j=0; j<k; j++){ 
                rk_fft_local[j] += obuff[j]; 
                // Exact correction for edges
                for(int l = j; l<k; l++){
                    rk[l+1] += (accumul_t)buff[len-j-1]*(accumul_t)buff[len-j+l];
                }
            }
            // Filling rk_fft_local beyond k
            for (; j<fftwlen; j++){
                rk_fft_local[j] += obuff[j];
            }
        }
        // Here's the optimization. Thanks to FFT's linearity!
        fftw_execute_r2r(rev_plan, rk_fft_local, obuff); // Reverse FFT
        // Manual reduction of ifft(rk_fft_local) to rk_mpfr
        #pragma omp critical
        for (int i=0; i<k; i++){
            rk_mpfr[i] += (mpreal)obuff[i]/(mpreal)fftwlen;
        }
        // Freeing memory
        fftw_free(ibuff);
        fftw_free(obuff);
        fftw_free(rk_fft_local);
    }
    // Leftover data! Probably too small to benefit from parallelization.
    for (uint64_t i=size-size%len; i<size; i++){
        m += (accumul_t)buffer[i];
        for (int j=0; j<k; j++){
            rk[j] += (accumul_t)buffer[i]*(accumul_t)buffer[i+j];
        }
    }
}

#endif
