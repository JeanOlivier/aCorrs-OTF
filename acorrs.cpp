#include "acorrs.h"
#include <fstream>

// Macro to generate C functions for U-type buffers, aka cheap C template.
#define gen_ACorrrUpTo(U, F) \
    static ACorrUpTo##F<U##_t>* aCorrUpTo##F##_ ## U = NULL;\
    \
    void ACorrUpTo##F##_##U##_destroy();\
    \
    void ACorrUpTo##F##_##U##_init FFT_ARGS(F){\
        ACorrUpTo##F##_##U##_destroy();\
        aCorrUpTo##F##_##U = new ACorrUpTo##F<U##_t>FFT_INST(F);\
    }\
    \
    void ACorrUpTo##F##_##U##_accumulate(U##_t *buffer, uint64_t size){\
        if (aCorrUpTo##F##_##U){\
            aCorrUpTo##F##_##U->accumulate(buffer, size);\
        }\
    }\
    \
    void ACorrUpTo##F##_##U##_accumulate_rk(U##_t *buffer, uint64_t size, double *res, int k){\
        if (aCorrUpTo##F##_##U){\
            aCorrUpTo##F##_##U->accumulate_m_rk(buffer, size);\
            aCorrUpTo##F##_##U->get_rk(res,k);\
        }\
    }\
    \
    void ACorrUpTo##F##_##U##_get_aCorrs(double* res, int k){\
        if (aCorrUpTo##F##_##U){\
            aCorrUpTo##F##_##U->get_aCorrs(res, k);\
        }\
    }\
    \
    void ACorrUpTo##F##_##U##_destroy(){\
        if (aCorrUpTo##F##_##U){\
            delete aCorrUpTo##F##_##U;\
            aCorrUpTo##F##_##U = NULL;\
        }\
    }

// Trick for different arguments in FFT case
#define FFT_ARGS(F) FFT_ARGS_##F
#define FFT_ARGS_FFT (int k, int len)
#define FFT_ARGS_ (int k)
#define FFT_INST(F) FFT_INST_##F
#define FFT_INST_FFT (k, len)
#define FFT_INST_ (k)


// Generating C functions
#ifdef __cplusplus
extern "C" {
#endif

// ACorrUpTo functions
gen_ACorrrUpTo(uint8,)
gen_ACorrrUpTo(int8,)
gen_ACorrrUpTo(uint16,)
gen_ACorrrUpTo(int16,)

// ACorrUpToFFT functions
gen_ACorrrUpTo(uint8, FFT)
gen_ACorrrUpTo(int8, FFT)
gen_ACorrrUpTo(uint16, FFT)
gen_ACorrrUpTo(int16, FFT)

// For setting desired mpreal precision beforehand
void set_mpreal_precision(int d){
    // Before any mreal are created
    const int digits = d; // Setting high precision
    mpreal::set_default_prec(mpfr::digits2bits(digits));
}

#ifdef __cplusplus
}
#endif


// Testing code  goes here. Not intended to be use as an executable.
int main(int argc, char *argv[]){
}
