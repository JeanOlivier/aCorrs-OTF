#include "acorrsPhi.hpp"

using namespace std;

template<class T>
inline ACorrUpToPhi<T>::ACorrUpToPhi(int k, int lambda): k(k), lambda(lambda), k_mpfr(k), l_mpfr(lambda), chunk_processed(0), block_processed(0)
{
    mf = new accumul_t [lambda]();      // Parentheses initialize to zero
    nfk = new uint64_t [lambda*k]();    // There are lambda phases and k lags -> lambda*k values
    rfk = new accumul_t [lambda*k](); 
    gfk = new accumul_t [lambda*k]();
    bfk = new accumul_t [lambda*k]();

    mf_mpfr = new mpreal [lambda]();
    Nfk_mpfr = new mpreal [lambda*k]();   
    rfk_mpfr = new mpreal [lambda*k]();   
    gfk_mpfr = new mpreal [lambda*k]();
    bfk_mpfr = new mpreal [lambda*k]();   

    aCorrs = new double [lambda*k]();
    aCorrs_mpfr = new mpreal [lambda*k]();   
    
    chunk_size = compute_chunk_size(); // Auto largest possible
}

// Methods //
template<class T>
inline void ACorrUpToPhi<T>::accumulate(T *buffer, uint64_t size){
    // On each call, a new block being processed.
    block_processed++;
    accumulate_Nfk(size);
    accumulate_gfk(buffer, size); // Compute bfk on very first data

    // 1. Do min(nfk) in blocks for all f and k with j-loop as the outer one; CHUNK
    // nfk[lambda*k-1] is min(nfk); nfk[f*k+j] >= nfk[lambda*k-1]
    // Loop on whole chunks
    uint64_t i; // Will point to last partial chunk after the loop 
    for (i=0; i<(nfk[lambda*k-1])/chunk_size; i++){
        accumulate_chunk(buffer+i*chunk_size, chunk_size);
        update(); // Update mpfr values and reset accumulators
    }
    // Last (potentially) partial chunk, keeps chunk_processed accurate
    if ((nfk[lambda*k-1])%chunk_size){
        accumulate_chunk(buffer+i*chunk_size, (nfk[lambda*k-1])%chunk_size);
        update(); // Update mpfr values and reset accumulatoris
    }
    // 2. Do the remainder from min(nfk) to nfk with j-loop as the inner one; EDGE
    // Right edge chunk, doesn't count in chunk_processed because it's small
    accumulate_chunk_edge(buffer, size);
    update(); // Update mpfr values and reset accumulators
    accumulate_bfk(buffer, size); // Computing (replacing) bfk on last data
}

template<class T>
inline void ACorrUpToPhi<T>::accumulate_chunk(T *buffer, uint64_t size){
    accumulate_mf_rfk(buffer, size); // Accumulating
    chunk_processed++;
}

template<class T>
inline void ACorrUpToPhi<T>::accumulate_chunk_edge(T *buffer, uint64_t size){
    accumulate_mf_rfk_edge(buffer, size); // Accumulating
    //chunk_processed++;
}

template<class T>
uint64_t ACorrUpToPhi<T>::get_nfk(uint64_t N, int lambda, int f, int k){
    // Whole block + Potential Partial Block - Avoid k out of buffer
    return N/lambda + ((N%lambda) > 0) - ((N%lambda+k+f)/lambda);
}

template<class T>
void ACorrUpToPhi<T>::compute_current_nfk(uint64_t size){
    for (int f=0; f<lambda; f++){
        for (int i=0; i<k; i++){
        nfk[f*k+i] = get_nfk(size, lambda, f, i);
        }
    }
}

template<class T>
void ACorrUpToPhi<T>::accumulate_Nfk(uint64_t size){
    compute_current_nfk(size);          // Set nfk to that of current block if not already done
    for (int f=0; f<lambda; f++){
        for (int i=0; i<k; i++){
            Nfk[f*k+i] += nfk[f*k+i];   // accumulating
        }
    }
}

// TODO: There's something fishy with j<size, look into it
template<class T>
inline void ACorrUpToPhi<T>::accumulate_mf_rfk(T *buffer, uint64_t size){
    #pragma omp parallel
    {
        manage_thread_affinity();
        #pragma omp for simd collapse(2) reduction(+:mf[:lambda]), reduction(+:rfk[:k*lambda])
        for (uint64_t j=0; j<size; j++){
            for (int f=0; f<lambda; f++){
                mf[f] += (accumul_t)buffer[f+j*lambda];
                #pragma omp ordered simd
                for (int i=0; i<k; i++){
                    rfk[f*k+i] += (accumul_t)buffer[f+j*lambda]*(accumul_t)buffer[f+j*lambda+i];
                }
            }
        }
        //// Test to minimise multiplications by lambda
        //#pragma omp for simd collapse(2) reduction(+:mf[:lambda]), reduction(+:rfk[:k*lambda])
        //for (uint64_t j=0; j<size*lambda; j+=lambda){
        //    for (int f=0; f<lambda; f++){
        //        mf[f] += (accumul_t)buffer[f+j];
        //        #pragma omp ordered simd
        //        for (int i=0; i<k; i++){
        //            rfk[f*k+i] += (accumul_t)buffer[f+j]*(accumul_t)buffer[f+j+i];
        //        }
        //    }
        //}
    }
}

template<class T>
inline void ACorrUpToPhi<T>::accumulate_mf_rfk_edge(T *buffer, uint64_t size){
    for (int i=0; i<k; i++){
        for (int f=0; f<lambda; f++){
            // Remainder of nfk, passed the common min(nfk)==nfk[lambda*k-1] 
            for (uint64_t j=nfk[lambda*k-1]; j<nfk[f*k+i]; j++){
                // Only counting once. Could probably be optmiized to avoid if
                if (i==0){
                    mf[f] += (accumul_t)buffer[f+j*lambda];
                }
                rfk[f*k+i] += (accumul_t)buffer[f+j*lambda]*(accumul_t)buffer[f+j*lambda+i];
            }
        }
    }
}

// gfk is the beginning corrections 
// It should be computed ONCE on the very first chunk of data
template<class T>
inline void ACorrUpToPhi<T>::accumulate_gfk(T *buffer, uint64_t size){
    for (int f=0; f<lambda; f++){
        uint64_t alphaf = size/lambda + (f<(int)(size%lambda));
        for (int i=0; i<k; i++){
            for (uint64_t j=0; j<alphaf-nfk[f*k+i]; j++){
                gfk[f*k+i] += (accumul_t)buffer[f+j*lambda];
            }
            gfk_mpfr[f*k+i] += gfk[f*k+i]; // Updating precision value
            gfk[f*k+i] = 0; // Reseting accumulator
        }
    }
}

// bfk is the end corrections
// Re-compute for each new chunk to keep autocorr valid 
template<class T>
inline void ACorrUpToPhi<T>::accumulate_bfk(T *buffer, uint64_t size){
    for (int f=0; f<lambda; f++){
        uint64_t alphaf = size/lambda + (f<(int)(size%lambda));
        for (int i=0; i<k; i++){
            for (uint64_t j=nfk[f*k+i]; j<alphaf; j++){
                bfk[f*k+i] += (accumul_t)buffer[f+j*lambda];
            }
            bfk_mpfr[f*k+i] += bfk[f*k+i]; // Updating precision value
            bfk[f*k+i] = 0; // Reseting accumulator
        }
    }
}
    
template<class T>
inline void ACorrUpToPhi<T>::update(){
    update_mpfr();
    reset_accumulators();
}

template<class T>
inline void ACorrUpToPhi<T>::update_mpfr(){
    for (int f=0; f<lambda; f++){
        mf_mpfr[f] += mf[f];
        for (int i=0; i<k; i++){
            Nfk_mpfr[f*k+i] += Nfk[f*k+i];
            rfk_mpfr[f*k+i] += rfk[f*k+i];
            // bfk/gfk accumulated in their own respective function
            //bfk_mpfr[f*k+i] += bfk[f*k+i];
            //gfk_mpfr[f*k+i] += gfk[f*k+i];
        }
    }
}

template<class T>
inline void ACorrUpToPhi<T>::reset_accumulators(){
    for (int f=0; f<lambda; f++){
        mf[f] = 0;
        for (int i=0; i<k; i++){
            Nfk[f*k+i] = 0;
            rfk[f*k+i] = 0;
            // bfk/gfk are reset in their own respective function
            //bfk[f*k+i] = 0;
            //gfk[f*k+i] = 0;
        }
    }
}

template<class T>
inline mpreal* ACorrUpToPhi<T>::get_aCorrs_mpfr(){
    // No corr across blocks: i -> i*block_processed
    for (int i=0; i<k; i++){
        for (int f=0; f<lambda; f++){
            aCorrs_mpfr[f*k+i] = (rfk_mpfr[f*k+i] - (mf_mpfr[f]-bfk_mpfr[f*k+i])*(mf_mpfr[f]-gfk_mpfr[f*k+i])/Nfk_mpfr[f*k+i])/Nfk_mpfr[f*k+i];
        }
    }
    return aCorrs_mpfr; // Return pointer to array
}

template<class T>
inline void ACorrUpToPhi<T>::compute_aCorrs(){
    get_aCorrs_mpfr();
    for (int l=0; l<k*lambda; l++){ // Single loop because there are no f-specific value
        aCorrs[l] = (double)aCorrs_mpfr[l]; // Small loss of precision here
    }
}

template<class T>
inline double* ACorrUpToPhi<T>::get_aCorrs(){
    compute_aCorrs();
    return aCorrs; // Return pointer to array
}

template<class T>
inline void ACorrUpToPhi<T>::get_aCorrs(double* res, int size){
    compute_aCorrs();
    // Size has to equal lambda*k
    for (int i=0; i<size; i++){
        res[i] = aCorrs[i];
    }
}

template<class T>
inline void ACorrUpToPhi<T>::get_rfk(double* res, int size){
        if (size>k){
            size = k;
        }
        // Size has to equal lambda*k
        for (int i=0; i<size; i++){
            res[i] = (double)rfk[i];
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
inline uint64_t ACorrUpToPhi<T>::compute_chunk_size(){
    uint64_t buff_max = max(abs(numeric_limits<T>::min()), abs(numeric_limits<T>::max()));
    uint64_t accumul_max = numeric_limits<accumul_t>::max();
    uint64_t ret = accumul_max/(buff_max*buff_max);
    return ret; // Int division removes fractional possibility
}


// Destructor //
template<class T>
inline ACorrUpToPhi<T>::~ACorrUpToPhi(){
    delete[] mf;
    delete[] nfk;
    delete[] rfk;
    delete[] gfk;
    delete[] bfk;
    delete[] mf_mpfr;
    delete[] Nfk_mpfr;
    delete[] rfk_mpfr;
    delete[] gfk_mpfr;
    delete[] bfk_mpfr;

    delete[] aCorrs;
    delete[] aCorrs_mpfr;
}

