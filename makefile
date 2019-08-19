# Toolchain, using mingw on windows under cywgin
CXX = $(OS:Windows_NT=x86_64-w64-mingw32-)g++
CP = cp
RM = rm
PY = $(OS:Windows_NT=/c/Anaconda2/)python

# flags
CFLAGS = -Ofast -march=native -std=c++11 -MMD -MP -Wall $(OS:Windows_NT=-DMS_WIN64 -D_hypot=hypot)
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

# directories
OBJ_DIR = obj
SRC_DIR = src
BIN_DIR = bin

# filenames
BIN := acorrs_wrapper
PYBIN := acorrs_otf.py
EXT := $(if $(filter $(OS),Windows_NT),pyd,so)
SRCS := $(shell find $(SRC_DIR) -name *.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
TARGET := $(BIN_DIR)/$(BIN).$(EXT)
PYTARGET := $(BIN_DIR)/$(PYBIN)


all: $(TARGET) $(PYTARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

# compile source
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp 
	$(CXX) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDLIBS) -c $< -o $@

# bring python files along
$(BIN_DIR)/%.py: $(SRC_DIR)/%.py
	$(CP) $< $@

clean:
	$(RM) -f $(OBJS) $(DEPS)
	$(RM) -f $(SRC_DIR)/*.pyc $(BIN_DIR)/*.pyc 

clean-all: clean
	[ -f $(TARGET) ] && $(RM) $(TARGET) || true 
	[ -f $(PYTARGET) ] && $(RM) $(PYTARGET) || true 

-include $(DEPS)

.PHONY: all clean clean-all
