#!/bin/bash
clear
rm -f testcase.o pairhmm_gpu.o main.cpp.o testcase_iterator.o libgkl_pairhmm.so
nvcc -c -arch=compute_52 -I. -I/usr/local/cuda/include ForwardDiagonalPipeline.cu -o pairhmm_gpu.o -std=c++11 -Wno-deprecated-gpu-targets -rdc=true -lcudadevrt -lcudart
nvcc -c --compiler-options -fPIC -arch=compute_52 -I. -I/usr/local/cuda/include ForwardDiagonalPipeline.cu -shared -o libgkl_pairhmm.so -std=c++11 -Wno-deprecated-gpu-targets -rdc=true -lcudadevrt -lcudart
g++ -Wall -W -std=c++11 -g -msse -mfpmath=sse -ffast-math -O4 -Wa,-q -pedantic -c testcase.cpp -o testcase.o
g++ -Wall -W -std=c++11 -g -msse -mfpmath=sse -ffast-math -O4 -Wa,-q -pedantic -c testcase_iterator.cpp -o testcase_iterator.o
g++ -c -l. GPUPairHMM.cc -o main.cpp.o -std=c++11 -g -msse -mfpmath=sse -ffast-math -O4
nvcc -o test pairhmm_gpu.o main.cpp.o testcase.o testcase_iterator.o -arch=compute_52 -std=c++11 -L/usr/local/cuda/lib -lcudart
