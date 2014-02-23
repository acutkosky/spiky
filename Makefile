
cuda_nef: cuda_nef.cu.h cuda_nef.cu testcudanef.cu
	nvcc -arch=sm_20 --compiler-options='-Wall' testcudanef.cu



