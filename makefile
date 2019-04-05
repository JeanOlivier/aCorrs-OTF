# Toolchain, using mingw on windows under cywgin
CC = $(OS:Windows_NT=x86_64-w64-mingw32-)g++
RM = rm
PY = $(OS:Windows_NT=/c/Anaconda2/)python

# flags
CFLAGS = -Ofast -march=native -std=c++11 -Wall $(OS:Windows_NT=-DMS_WIN64 -D_hypot=hypot)
OMPFLAGS = -fopenmp -fopenmp-simd
SHRFLAGS = -fPIC -shared
FFTWFLAGS = -lfftw3 -lm

# includes
PYINCL = `$(PY) -m pybind11 --includes`
ifneq ($(OS),Windows_NT)
    PYINCL += -I /usr/include/python2.7/
endif
    


# libraries
LDLIBS = -lmpfr $(OS:Windows_NT=-L /c/Anaconda2/libs/ -l python27) $(PYINCL) 


# filenames
HEAD = acorrs.h
SRC = acorrs_wrapper.cpp
EXT = $(if $(filter $(OS),Windows_NT),.pyd,.so)
TARGET = $(SRC:.cpp=$(EXT))


all: $(TARGET)

$(TARGET): $(SRC) $(HEAD)
	$(CC) $(SRC) -o $(TARGET) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

force: 
	$(CC) $(SRC) -o $(TARGET) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

clean:
	@[ -f $(TARGET) ] && $(RM) $(TARGET) || true

.PHONY: all clean force 
