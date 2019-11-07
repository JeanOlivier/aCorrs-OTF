#ifndef autoco_H
#define autoco_H

#include "common.hpp"

//TODO: Make (it?) a general correlation class (with aCorr as a special case?)
template<class T> class ACorrUpTo
{
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
    ACorrUpTo(int k);

// Methods //
    void accumulate(T *buffer, uint64_t size);
    inline void accumulate_chunk(T *buffer, uint64_t size);
    inline void accumulate_chunk_edge(T *buffer, uint64_t size);
    virtual void accumulate_m_rk(T *buffer, uint64_t size);
    inline void accumulate_m_rk_edge(T *buffer, uint64_t size);
    // gk is the beginning corrections 
    // It should be computed ONCE on the very first chunk of data
    inline void accumulate_gk(T *buffer, uint64_t size);
    // bk is the end corrections
    // Re-compute for each new chunk to keep autocorr valid 
    inline void accumulate_bk(T *buffer, uint64_t size);

    inline void update();
    inline void update_mpfr();
    inline void reset_accumulators();

    mpreal get_mean_mpfr();
    double get_mean();
    mpreal get_var_mpfr();
    double get_var();
    mpreal* get_aCorrs_mpfr();
    void compute_aCorrs();
    double* get_aCorrs();
    void get_aCorrs(double* res, int size);
    void get_rk(double* res, int size);
   
    uint64_t compute_chunk_size();

// Destructor //
   virtual ~ACorrUpTo();
};


#endif // autoco_H
