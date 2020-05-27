#include "acorrsFFT.hpp"

using namespace std;

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
inline ACorrUpToFFT<T>::ACorrUpToFFT(int k, int len): ACorrUpTo<T>(k), len(len)
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

    // Max number of double accumulation before we get errors on correlations
    counter_max = compute_accumul_max();
}

/*
The number of time we accumulate in Fourier space is limited for accuracy.
1. A double can exactly represent successive integers -2^53:2^53
2. After the FFT roundtrip, we have an int*int times the int fftwlen, an integer
3. If the accumulator reaches beyond this, we create errors no matter what
4. Standard double accumulation errors also occurs
5. To mitigate those, we cap the number of accumulations to a reasonable value of 4096
6. Testing with 2 GiSa yields rk rel. errors for {int8, uint8, int16, uint16} smaller than:
     {0, 0, 5e-15, 9e-16} for worse case scenario of max amplitude values
     {0, 0,    ±0, 2e-16} for random uniforms
   This is better or similar to the error induced when casting the final result to double
   It is thus considered acceptable
7. Kahan summation doesn't seem to help much and slows things down
8. 3/4 factor added empirically to avoid errors on rk[0]
*/
template <class T>
inline int ACorrUpToFFT<T>::compute_accumul_max(){
    uint64_t buff_max = max(abs(numeric_limits<T>::min()), abs(numeric_limits<T>::max()));
    int ret = (int)min((((uint64_t)1)<<53)/((uint64_t)fftwlen*buff_max*buff_max),(uint64_t)4096)*3/4;
    return ret;
}

template <class T>
inline ACorrUpToFFT<T>::~ACorrUpToFFT(){
    // Saving wisdom for future use
    fftw_export_wisdom_to_filename("FFTW_Wisdom.dat");
    // Deleting plans
    fftw_destroy_plan(fwd_plan);
    fftw_destroy_plan(rev_plan);
    // Deleting temp buffers
    fftw_free(in); 
    fftw_free(out);
}

template <class T> 
inline void ACorrUpToFFT<T>::accumulate_m_rk(T *buffer, uint64_t size){
    uint64_t fftnum = size/len;
    #pragma omp parallel
    {
        manage_thread_affinity();
        double *ibuff = fftw_alloc_real(fftwlen);
        double *obuff = fftw_alloc_real(fftwlen);
        double *rk_fft_local = fftw_alloc_real(fftwlen);
        for (int i=0; i<fftwlen; i++){
            rk_fft_local[i] = 0;
        }

        // Each thread accumulates at most counter_max iterations to minimize errors
        int counter = 0;

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
                for(int l = j; l<k-1; l++){
                    rk[l+1] += (accumul_t)buff[len-j-1]*(accumul_t)buff[len-j+l];
                }
            }
            // Filling rk_fft_local beyond k
            for (; j<fftwlen; j++){
                rk_fft_local[j] += obuff[j]; 
            }
            counter++;
            if (counter==counter_max){
                counter = 0;
                // Here's the optimization. Thanks to FFT's linearity!
                fftw_execute_r2r(rev_plan, rk_fft_local, obuff); // Reverse FFT
                // Manual reduction of ifft(rk_fft_local) to rk_mpfr
                #pragma omp critical
                for (int i=0; i<k; i++){
                    // rk_mpfr would be an integer if not for floating point errors 
                    // outter round might be sufficient, rint_round rounds .5 away from 0
                    rk_mpfr[i] += mpfr::rint_round(mpfr::rint_round((mpreal)obuff[i])/(mpreal)fftwlen); 
                }
                // Zeroing for next iteration
                for (int i=0; i<fftwlen; i++){
                    rk_fft_local[i] = 0;
                }
            }
        }
        if (counter != 0){
            // Here's the optimization. Thanks to FFT's linearity!
            fftw_execute_r2r(rev_plan, rk_fft_local, obuff); // Reverse FFT
            // Manual reduction of ifft(rk_fft_local) to rk_mpfr
            #pragma omp critical
            for (int i=0; i<k; i++){
                // rk_mpfr would be an integer if not for floating point errors 
                // outter round might be sufficient, rint_round rounds .5 away from 0
                rk_mpfr[i] += mpfr::rint_round(mpfr::rint_round((mpreal)obuff[i])/(mpreal)fftwlen); 
            }
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

