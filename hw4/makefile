CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CYY = nvcc
CYYFLAGS = -std=c++11 -O3 -Xcompiler -fopenmp  -arch=sm_61


# turning off auto-vectorization since this can make hand-vectorized code slower
CXXFLAGS += -fno-tree-vectorize

TARGETS = $(basename $(wildcard *.cpp))
TARGETS1 = $(basename $(wildcard *.cu))

all : $(TARGETS)
all : $(TARGETS1)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.cu *.h
	$(CYY) $(CYYFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
