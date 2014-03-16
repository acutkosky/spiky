
cuda_nef: cuda_nef.cu.h cuda_nef.cu testcudanef.cu
	nvcc -arch=sm_20 --compiler-options='-Wall' testcudanef.cu

mnist_nef: cuda_nef.cu.h cuda_nef.cu MNIST_cuda_nef.cu read_mnist.cpp
	nvcc -arch=sm_20 --compiler-options='-Wall' MNIST_cuda_nef.cu -o MNIST_cuda


