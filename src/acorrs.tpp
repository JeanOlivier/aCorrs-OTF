#include "acorrs.hpp"

using namespace std;

template<class T>
inline ACorrUpTo<T>::ACorrUpTo(int k): m(0), n(0), k(k), m_mpfr(0), n_mpfr(0), k_mpfr(k)
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
template<class T>
inline void ACorrUpTo<T>::accumulate(T *buffer, uint64_t size){
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

template<class T>
inline void ACorrUpTo<T>::accumulate_chunk(T *buffer, uint64_t size){
    accumulate_m_rk(buffer, size); // Accumulating
    chunk_processed++;
}

template<class T>
inline void ACorrUpTo<T>::accumulate_chunk_edge(T *buffer, uint64_t size){
    accumulate_m_rk_edge(buffer, size); // Accumulating
    //chunk_processed++;
}
template<class T>
inline void ACorrUpTo<T>::accumulate_m_rk(T *buffer, uint64_t size){
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

template<class T>
inline void ACorrUpTo<T>::accumulate_m_rk_edge(T *buffer, uint64_t size){
    for (uint64_t i=size-k; i<size; i++){
        m += (accumul_t)buffer[i];
        for (uint64_t j=0; j<size-i; j++){
            rk[j] += (accumul_t)buffer[i]*(accumul_t)buffer[i+j];
        }
    }
}

// gk is the beginning corrections 
// It should be computed ONCE on the very first chunk of data
template<class T>
inline void ACorrUpTo<T>::accumulate_gk(T *buffer, uint64_t size){
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
template<class T>
inline void ACorrUpTo<T>::accumulate_bk(T *buffer, uint64_t size){
    for (int i=0; i<k; i++){
        for (uint64_t j=size-i; j<size; j++){
            bk[i] += (accumul_t)buffer[j];
        }
        bk_mpfr[i] += bk[i]; // Updating precision value
        bk[i] = 0; // Reseting accumulator
    }
}
    
template<class T>
inline void ACorrUpTo<T>::update(){
    update_mpfr();
    reset_accumulators();
}

template<class T>
inline void ACorrUpTo<T>::update_mpfr(){
    m_mpfr += m;        
    n_mpfr += n;
    for (int i=0; i<k; i++){
        rk_mpfr[i] += rk[i]; 
    } // bk_mpfr and gk_mpfr are updated at computation
}

template<class T>
inline void ACorrUpTo<T>::reset_accumulators(){
    m=0;
    n=0;
    for (int i=0; i<k; i++){
        rk[i] = 0;
    }
}

template<class T>
inline mpreal ACorrUpTo<T>::get_mean_mpfr(){
    update(); // Just to be sure
    mpreal r = m_mpfr/n_mpfr;
    return r;
}

template<class T>
inline double ACorrUpTo<T>::get_mean(){
    return (double)get_mean_mpfr();
}

template<class T>
inline mpreal ACorrUpTo<T>::get_var_mpfr(){
    update(); // Just to be sure
    mpreal v = (rk_mpfr[0]-pow(m_mpfr,2)/n_mpfr)/(n_mpfr);
    return v;
}

template<class T>
inline double ACorrUpTo<T>::get_var(){
    return (double)get_var_mpfr();
}

template<class T>
inline mpreal* ACorrUpTo<T>::get_aCorrs_mpfr(){
    // No corr across blocks: i -> i*block_processed
    mpreal n_k;
    for (int i=0; i<k; i++){
        n_k = n_mpfr - (mpreal)(i*block_processed);
        aCorrs_mpfr[i] = (rk_mpfr[i] - (m_mpfr-bk_mpfr[i])*(m_mpfr-gk_mpfr[i])/n_k)/n_k;
    }
    return aCorrs_mpfr; // Return pointer to array
}

template<class T>
inline void ACorrUpTo<T>::compute_aCorrs(){
    get_aCorrs_mpfr();
    for (int i=0; i<k; i++){
        aCorrs[i] = (double)aCorrs_mpfr[i]; // Small loss of precision here
    }
}

template<class T>
inline double* ACorrUpTo<T>::get_aCorrs(){
    compute_aCorrs();
    return aCorrs; // Return pointer to array
}

template<class T>
inline void ACorrUpTo<T>::get_aCorrs(double* res, int size){
    compute_aCorrs();
    for (int i=0; i<size; i++){
        res[i] = aCorrs[i];
    }
}

template<class T>
inline void ACorrUpTo<T>::get_rk(double* res, int size){
        if (size>k){
            size = k;
        }
        for (int i=0; i<size; i++){
            res[i] = (double)rk[i];
        }
    }

   
// Max chunk_size to avoid overflow: chunk_size*buff_max² == accumul_max
// Relevant quantities are max(accumul_t) and max(abs(min(buff)), max(buff))
// e.g. int16 buff spanning -2^15:2^15-1 == -32768:32767 in int64 accumulator:
//   - max positive buff squared value is (-2^15)^2 = 2^30 = 1073741824
//   - max negative buff squared value is -2^15*(2^15-1) = -2^30+2^15 = -1073709056
//   - accumulator max positive value is 2^63-1 -> (2^63-1)/2^30 = (2^33 - 1) + (1 - 2^-30)
//       - With result stated as a positive sum of the integer and fractional parts
//       - Casting back to int64 yields 2^33-1 = 8589934591
//   - accumulator max negative value is -2^63 
//          -> -2^63/(-2^15*(2^15-1) = 2^33/(1-2^-15) = (2^33+2**18)/(1-2**-30) 
//          -> 2^33 + 2^18 + epsilon (first order Taylor, positive epsilon tends to 0)
//       - Casting back to int64 yields 2^33 + 2^18 + 0 = 8590196736 > 8589934591
//   - The chunk_size is thus the smallest results: 8589934591
// e.g. uint16 spanning 0:2^16-1 in uint64 accumulator
//   - Similarly: 2^64-1/(2^16-1)^2 = 2^32 + 2^17 - 1 + 4 + epsilon -> 4295098371
template<class T>
inline uint64_t ACorrUpTo<T>::compute_chunk_size(){
    uint64_t buff_max = max(abs(numeric_limits<T>::min()), abs(numeric_limits<T>::max()));
    uint64_t accumul_max = numeric_limits<accumul_t>::max();
    uint64_t ret = accumul_max/(buff_max*buff_max);
    return ret; // Int division removes fractional possibility
}


// Destructor //
template<class T>
inline ACorrUpTo<T>::~ACorrUpTo(){
    delete[] rk;
    delete[] gk;
    delete[] bk;
    delete[] rk_mpfr;
    delete[] gk_mpfr;
    delete[] bk_mpfr;
}

