#ifndef autocoPhi_H
#define autocoPhi_H

#include "common.hpp"

//TODO: Make (it?) a general correlation class (with aCorr as a special case?)
template<class T> class ACorrUpToPhi
{
public:
// Variables //
    
    // Casting over different type sign is very slow, we avoid it.
    // (T)-1>0 is true only if T is unsigned
    // Would worfk with int < 64bits but would overflow faster
    typedef typename conditional<((T)-1>0), uint64_t, int64_t>::type accumul_t;
    
    // Math stuff 
    //accumul_t n;  // Actual length of data
    uint64_t *nfk;  // nfk for current block
    int k;
    int lambda;     // Period, phases will span [0, lambda-1]

    // f,k specific accumulators
    accumul_t *mf;  // One m per phase
    accumul_t *rfk; // Conceptually: rfk[i][j] == rfk[i*k+j]
    accumul_t *bfk; // Similarly
    accumul_t *gfk; // Similarly
    
    // Precision stuff
    mpreal k_mpfr;
    mpreal l_mpfr;

    mpreal *mf_mpfr;
    mpreal *Nfk_mpfr;
    mpreal *rfk_mpfr;
    mpreal *bfk_mpfr;
    mpreal *gfk_mpfr;

    // Autocorrelations results
    mpreal *aCorrs_mpfr;
    double *aCorrs;

    // Managerial stuff
    uint64_t chunk_processed;
    uint64_t chunk_size;
    uint64_t block_processed;

// Constructors //
    ACorrUpToPhi(int k, int lambda);

// Methods //
    uint64_t get_nfk(uint64_t N, int lambda, int f, int k);
    void compute_current_nfk(uint64_t size);
    void accumulate_Nfk(uint64_t size);   // Denominator for f and k
    void accumulate(T *buffer, uint64_t size);
    inline void accumulate_chunk(T *buffer, uint64_t size);
    inline void accumulate_chunk_edge(T *buffer, uint64_t size);
    virtual void accumulate_mf_rfk(T *buffer, uint64_t nphi);
    inline void accumulate_mf_rfk_edge(T *buffer, uint64_t size);
    // gfk is the beginning corrections 
    // It should be computed ONCE on the very first chunk of data
    inline void accumulate_gfk(T *buffer, uint64_t size);
    // bfk is the end corrections
    // Re-compute for each new chunk to keep autocorr valid 
    inline void accumulate_bfk(T *buffer, uint64_t size);

    inline void update();
    inline void update_mpfr();
    inline void reset_accumulators();

    mpreal* get_aCorrs_mpfr();
    void compute_aCorrs();
    double* get_aCorrs();
    void get_aCorrs(double* res, int size);
    void get_rfk(double* res, int size);
   
    uint64_t compute_chunk_size();

// Destructor //
   virtual ~ACorrUpToPhi();
};


#endif // autocoPhi_H
