#include <cuda_runtime.h>

// so the __global__ kernel has a few automatically defined variables available to it, including:
// blockIdx, blockDim, threadIdx, gridDim, etc
// have (.x, .y, .z) components for 3D indexing
// __sync_threads() waits for all threads in a block to reach to the same point
// warp and other things might be a bit more advanced
// essentially special CUDA variables, like how we have keywords like "this" in c++

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < N) C[i] = A[i] + B[i];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
   int threadsPerBlock = 256;
   int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

   vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
   cudaDeviceSynchronize();
}