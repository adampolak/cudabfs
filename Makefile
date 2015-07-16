NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-arch sm_50 -O3

MGPU=3rdparty/moderngpu
MGPUFLAGS=-I $(MGPU)/include $(MGPU)/src/mgpucontext.cu $(MGPU)/src/mgpuutil.cpp

CXX=g++
CXXFLAGS=-O3

all: bfs-mgpu.e dimacs-parser.e

bfs-mgpu.e: bfs-mgpu.cu
	$(NVCC) $(NVCCFLAGS) $(MGPUFLAGS) $< -o $@

dimacs-parser.e: dimacs-parser.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.e

.PHONY: all clean
