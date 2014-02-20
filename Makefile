
cuda_nef: cuda_nef.cpp cuda_nef.h testcudanef.cpp
	nvcc -std=gnu++11 testcudanef.cpp -o testcudanef


