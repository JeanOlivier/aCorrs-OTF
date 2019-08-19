#ifndef common_H
#define common_H

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
#include <math.h>

#include <iostream>
#include <iomanip>

#include <omp.h>
#include <limits>


#ifdef _WIN32_WINNT
    #include "mpreal.h"
#else
    #include <mpreal.h>
#endif


#define __STDC_FORMAT_MACROS
#include <inttypes.h>


using namespace std;
using mpfr::mpreal;

// For setting desired mpreal precision beforehand
void set_mpreal_precision(int d);

void manage_thread_affinity();


#endif // common_H
