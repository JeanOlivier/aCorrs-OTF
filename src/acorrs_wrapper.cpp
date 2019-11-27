#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <string.h>
#include "acorrs.hpp"
#include "acorrsFFT.hpp"
#include "acorrsPhi.hpp"


namespace py = pybind11;

//TODO: Minimize redundant code by somehow integrating both declaration classes?


template<typename T>
void declare_class(py::module &m, std::string typestr) {
    using Class = ACorrUpTo<T>;
    std::string pyclass_name = std::string("ACorrUpTo_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<int>())
        .def("accumulate", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            }
        )
        .def("accumulate_m_rk", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate_m_rk((T*)buff.ptr, buff.size);
            }
        )        
        .def("compute_aCorrs", &Class::compute_aCorrs)
        .def("get_aCorrs", [](Class& self) {
            return py::array_t<double>(
                {self.k,},          // shape
                {sizeof(double),},  // C-style contiguous strides for double
                self.aCorrs,        // the data pointer
                NULL);              // numpy array references this parent
            }
        )
        .def("__call__", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            self.compute_aCorrs();
            }
        )
        
        .def_property_readonly("res", [](Class& self){
            double *tmp; 
            if (self.n){
                tmp = self.get_aCorrs();
            }
            else {
                tmp = self.aCorrs;
            }
            return py::array_t<double>(
                {self.k,},          // shape
                {sizeof(double),},  // C-style contiguous strides for double
                tmp,                // the data pointer
                NULL);              // numpy array references this parent
            }
        )
        .def_property_readonly("rk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.rk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.rk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("bk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.bk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.bk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("gk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.gk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.gk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("k", [](Class& self) {return self.k;})
        .def_property_readonly("m", [](Class& self) {return (decltype(self.m)) self.m_mpfr;})
        .def_property_readonly("n", [](Class& self) {return (decltype(self.n)) self.n_mpfr;})
        .def_property_readonly("chunk_processed", [](Class& self) {return self.chunk_processed;})
        .def_property_readonly("chunk_size", [](Class& self) {return self.chunk_size;})
        .def_property_readonly("block_processed", [](Class& self) {return self.block_processed;})
        ;
}


template<typename T>
void declare_fftclass(py::module &m, std::string typestr) {
    using Class = ACorrUpToFFT<T>;
    std::string pyclass_name = std::string("ACorrUpToFFT_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<int,int>())
        .def("accumulate", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            }
        )
        .def("accumulate_m_rk", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate_m_rk((T*)buff.ptr, buff.size);
            }
        )
        .def("compute_aCorrs", &Class::compute_aCorrs)
        .def("get_aCorrs", [](Class& self) {
            return py::array_t<double>(
                {self.k,},          // shape
                {sizeof(double),},  // C-style contiguous strides for double
                self.aCorrs,        // the data pointer
                NULL);              // numpy array references this parent
            }
        )
        .def("__call__", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            self.compute_aCorrs();
            }
        )
        
        .def_property_readonly("res", [](Class& self){
            double *tmp; 
            if (self.n){
                tmp = self.get_aCorrs();
            }
            else {
                tmp = self.aCorrs;
            }
            return py::array_t<double>(
                {self.k,},          // shape
                {sizeof(double),},  // C-style contiguous strides for double
                tmp,                // the data pointer
                NULL);              // numpy array references this parent
            }
        )
        .def_property_readonly("rk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.rk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.rk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("bk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.bk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.bk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("gk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.gk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.gk_mpfr[i];}
            return py::array_t<accumul_t>(
                {self.k,},          // shape
                {sizeof(accumul_t),},  // C-style contiguous strides for double
                ret,                // the data pointer
                free_when_done);    // numpy array references this parent
            }
        )
        .def_property_readonly("k", [](Class& self) {return self.k;})
        // Using the type of n instead of m because of &m voodoo. Always the same. 
        .def_property_readonly("m", [](Class& self) {return (decltype(self.n)) self.m_mpfr;})
        .def_property_readonly("n", [](Class& self) {return (decltype(self.n)) self.n_mpfr;})
        .def_property_readonly("len", [](Class& self) {return self.len;})
        .def_property_readonly("fftwlen", [](Class& self) {return self.fftwlen;})
        .def_property_readonly("chunk_processed", [](Class& self) {return self.chunk_processed;})
        .def_property_readonly("chunk_size", [](Class& self) {return self.chunk_size;})
        .def_property_readonly("block_processed", [](Class& self) {return self.block_processed;})
        .def_property_readonly("counter_max", [](Class& self) {return self.counter_max;})
        ;
}

template<typename T>
void declare_phiclass(py::module &m, std::string typestr) {
    using Class = ACorrUpToPhi<T>;
    std::string pyclass_name = std::string("ACorrUpToPhi_") + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
        .def(py::init<int,int>())
        .def("get_nfk", [](Class& self, uint64_t a, int b, int c, int d){
            return self.get_nfk(a,b,c,d);
            }
        )
        .def("compute_current_nfk", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            self.compute_current_nfk(buff.size);
            auto res = py::array_t<uint64_t>(
                {self.lambda*self.k,},  // shape
                {sizeof(uint64_t),},      // C-style contiguous strides for double
                self.nfk,            // the data pointer
                NULL);                  // numpy array references this parent
            res.resize({self.lambda, self.k});
            return res;
            }
        )
        .def("accumulate", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            }
        )
        .def("accumulate_mf_rfk", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate_mf_rfk((T*)buff.ptr, buff.size);
            }
        )
        .def("compute_aCorrs", &Class::compute_aCorrs)
        .def("get_aCorrs", [](Class& self) {
            auto res = py::array_t<double>(
                {self.lambda*self.k,},  // shape
                {sizeof(double),},      // C-style contiguous strides for double
                self.aCorrs,            // the data pointer
                NULL);                  // numpy array references this parent
            res.resize({self.lambda, self.k});
            return res;
            }
        )
        .def("__call__", [](Class& self, py::array_t<T, py::array::c_style>& array) {
            auto buff = array.request();
            pybind11::gil_scoped_release release;   
            self.accumulate((T*)buff.ptr, buff.size);
            self.compute_aCorrs();
            }
        )
        
        .def_property_readonly("res", [](Class& self){
            double *tmp; 
            if (self.nfk[0]){
                tmp = self.get_aCorrs();
            }
            else {
                tmp = self.aCorrs;
            }
            auto res = py::array_t<double>(
                {self.lambda*self.k,},  // shape
                {sizeof(double),},      // C-style contiguous strides for double
                tmp,                    // the data pointer
                NULL);                  // numpy array references this parent
            res.resize({self.lambda, self.k});
            return res;
            }
        )
        .def_property_readonly("res0", [](Class& self){
            self.get_aCorrs0();
            return py::array_t<double>(
                {self.k,},  // shape
                {sizeof(double),},      // C-style contiguous strides for double
                self.ak,                    // the data pointer
                NULL);                  // numpy array references this parent
            }
        )
        .def_property_readonly("rk", [](Class& self){
            typedef typename std::remove_pointer<decltype(self.rfk)>::type accumul_t;
            accumul_t *ret = new accumul_t [self.lambda*self.k]();
            py::capsule free_when_done(ret, [](void *f) {
                accumul_t *ret = reinterpret_cast<accumul_t *>(f);
                delete[] ret;
                });
            for (int i=0; i<self.lambda*self.k; i++){ret[i] = (accumul_t)self.rfk_mpfr[i];}
            auto res = py::array_t<accumul_t>(
                {self.lambda*self.k,}, // shape
                {sizeof(accumul_t),},   // C-style contiguous strides for double
                ret,                    // the data pointer
                free_when_done);        // numpy array references this parent
            res.resize({self.lambda, self.k});
            return res;
            }
        )
        //.def_property_readonly("bk", [](Class& self){
        //    typedef typename std::remove_pointer<decltype(self.bk)>::type accumul_t;
        //    accumul_t *ret = new accumul_t [self.k]();
        //    py::capsule free_when_done(ret, [](void *f) {
        //        accumul_t *ret = reinterpret_cast<accumul_t *>(f);
        //        delete[] ret;
        //        });
        //    for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.bk_mpfr[i];}
        //    return py::array_t<accumul_t>(
        //        {self.k,},          // shape
        //        {sizeof(accumul_t),},  // C-style contiguous strides for double
        //        ret,                // the data pointer
        //        free_when_done);    // numpy array references this parent
        //    }
        //)
        //.def_property_readonly("gk", [](Class& self){
        //    typedef typename std::remove_pointer<decltype(self.gk)>::type accumul_t;
        //    accumul_t *ret = new accumul_t [self.k]();
        //    py::capsule free_when_done(ret, [](void *f) {
        //        accumul_t *ret = reinterpret_cast<accumul_t *>(f);
        //        delete[] ret;
        //        });
        //    for (int i=0; i<self.k; i++){ret[i] = (accumul_t)self.gk_mpfr[i];}
        //    return py::array_t<accumul_t>(
        //        {self.k,},          // shape
        //        {sizeof(accumul_t),},  // C-style contiguous strides for double
        //        ret,                // the data pointer
        //        free_when_done);    // numpy array references this parent
        //    }
        //)
        .def_property_readonly("k", [](Class& self) {return self.k;})
        .def_property_readonly("n", [](Class& self) {return self.n;}) 
        .def_property_readonly("l", [](Class& self) {return self.lambda;})
        .def_property_readonly("chunk_processed", [](Class& self) {return self.chunk_processed;})
        .def_property_readonly("chunk_size", [](Class& self) {return self.chunk_size;})
        .def_property_readonly("block_processed", [](Class& self) {return self.block_processed;})
        ;
}


#define declare_class_for(U) declare_class<U##_t>(m, std::string(#U));
#define declare_fftclass_for(U) declare_fftclass<U##_t>(m, std::string(#U));
#define declare_phiclass_for(U) declare_phiclass<U##_t>(m, std::string(#U));

PYBIND11_MODULE(acorrs_wrapper, m) {
    m.doc() = "pybind11 wrapper for acorrs.h"; // optional module docstring
    m.attr("the_answer") = 42;
    m.def("set_mpreal_precision", &set_mpreal_precision);

    declare_class_for(uint8)
    declare_class_for(int8)
    declare_class_for(uint16)
    declare_class_for(int16)
    
    declare_fftclass_for(uint8)
    declare_fftclass_for(int8)
    declare_fftclass_for(uint16)
    declare_fftclass_for(int16)
    
    declare_phiclass_for(uint8)
    declare_phiclass_for(int8)
    declare_phiclass_for(uint16)
    declare_phiclass_for(int16)
}

