# Toolchain, using mingw on windows
CC = $(OS:Windows_NT=x86_64-w64-mingw32-)g++
PKG_CFG = $(OS:Windows_NT=x86_64-w64-mingw32-)pkg-config
RM = rm

# flags
CFLAGS = -Ofast -march=native -Wall
OMPFLAGS = -fopenmp -fopenmp-simd
SHRFLAGS = -fPIC -shared
FFTWFLAGS = -lfftw3 -lm

# libraries
LDLIBS = -lmpfr

# filenames
HEAD = acorrs.h
SRC = acorrs.cpp
EXT = $(if $(filter $(OS),Windows_NT),.exe,.out)
TARGET = $(SRC:.cpp=$(EXT))
SHREXT = $(if $(filter $(OS),Windows_NT),.dll,.so)
SHRTGT = $(SRC:.cpp=$(SHREXT))


all: $(SHRTGT) $(TARGET)

$(SHRTGT): $(SRC) $(HEAD)
	$(CC) $(SRC) -o $(SHRTGT) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

$(TARGET): $(SRC) $(HEAD)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

force: 
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)
	$(CC) $(SRC) -o $(SHRTGT) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(FFTWFLAGS) $(LDLIBS)

clean:
	@[ -f $(TARGET) ] && $(RM) $(TARGET) || true
	@[ -f $(SHRTGT) ] && $(RM) $(SHRTGT) || true

.PHONY: all clean force 
