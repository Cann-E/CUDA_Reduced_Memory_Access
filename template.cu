#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A: m x k, B: k x n, C: m x n (row-major)

// Kernel 1: naive row-indexing
__global__ void matmul1_naive(float *A, float *B, float *C, int M, int N, int K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // column index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// Kernel 2: coalesced indexing (swapped thread indices)
__global__ void matmul2_coalesced(float *A, float *B, float *C, int M, int N, int K) {
    int j = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // row index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// kernel 3: coalesced, coarsened 2x2
// TODO: develop your code here based on Kernel 2.
// Add thread coarsening: start with 2x2, meaning each thread now is
// responsible to compute 4 elements (a 2x2 patch) of C.
// Example GFLOPS if successful: ~3000 GFLOPS for matrix size 4096
__global__ void coarsened_matmul2x2(float *A, float *B, float *C, int M, int N, int K)
{
    //thread  with coarsening 2x2
    int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2; 
    int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2; 

    if (i >= M || j >= N) return;

    // Registers for 2x2 
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    for (int k = 0; k < K; k++) {
        
        float a0 = A[i * K + k];                         
        float a1 = (i + 1 < M) ? A[(i + 1) * K + k] : 0; 
        
        
        float b0 = B[k * N + j];                         
        float b1 = (j + 1 < N) ? B[k * N + (j + 1)] : 0; 

        //  2x2 block of C
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }

    
    C[i * N + j] = c00;
    if (j + 1 < N) C[i * N + (j + 1)] = c01;
    if (i + 1 < M) C[(i + 1) * N + j] = c10;
    if (i + 1 < M && j + 1 < N) C[(i + 1) * N + (j + 1)] = c11;
}


// Kernel 4: shared memory (SMEM) tiled
// TODO: develop your code here. You should make use of shared memory
// and Tiling technique to reduce global memory access. 
// Each thread block computes a TSxTS block of C, where
// A is MxK, B is KxN, and C is MxN (all row‑major).
// Each thread is still responsible to compute one element in C
#define TS 16  

__global__ void MatMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float Asub[TS][TS];  
    __shared__ float Bsub[TS][TS];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TS + ty;
    int col = blockIdx.x * TS + tx;
    float c = 0.0f;

    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        if (row < M && (t * TS + tx) < K) 
            Asub[ty][tx] = A[row * K + t * TS + tx];
        else 
            Asub[ty][tx] = 0;

        if (col < N && (t * TS + ty) < K) 
            Bsub[ty][tx] = B[(t * TS + ty) * N + col];
        else 
            Bsub[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < TS; k++) 
            c += Asub[ty][k] * Bsub[k][tx];

        __syncthreads();
    }

    if (row < M && col < N) 
        C[row * N + col] = c;
}



// Kernel 5: shared memory (SMEM) tiled
// TODO: Do your best to have the fastest correct kernel here.
// Things to do (might be a good idea in this order):
// 1. add thread coarsening to previous tiling kernel: for example,
//    your thread block dim can be 16x16 (256 threads per block), yet
//    it computes a 32x32 matrix block in C. Each thread computes 2x2 elements
//    in C.
//    Example GFLOPS: ~4000 GFLOPS. 
// 2. increase the coarsening factor to 4x4, 8x8 etc.
//    Example GFLOPS: 4x4 ~8000 GFLOPS, 8x8 ~10000 GFLOPS
// 3. Tuning parameters: block dim, coarsening factor, ... 
#define BS 16
#define COARSE 4

__global__ void MatmulBest(float *A, float *B, float *C, int M, int N, int K) {
    int row_tile = blockIdx.y * BS * COARSE;
    int col_tile = blockIdx.x * BS * COARSE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float accum[COARSE][COARSE];
    for (int i = 0; i < COARSE; i++)
        for (int j = 0; j < COARSE; j++)
            accum[i][j] = 0.0f;
    __shared__ float As[BS * COARSE][BS];
    __shared__ float Bs[BS][BS * COARSE];
    int numTiles = (K + BS - 1) / BS;
    for (int t = 0; t < numTiles; t++) {
        for (int i = 0; i < COARSE; i++) {
            int rowA = row_tile + ty * COARSE + i;
            int colA = t * BS + tx;
            if (rowA < M && colA < K)
                As[ty * COARSE + i][tx] = A[rowA * K + colA];
            else
                As[ty * COARSE + i][tx] = 0.0f;
        }
        for (int j = 0; j < COARSE; j++) {
            int rowB = t * BS + ty;
            int colB = col_tile + tx * COARSE + j;
            if (rowB < K && colB < N)
                Bs[ty][tx * COARSE + j] = B[rowB * N + colB];
            else
                Bs[ty][tx * COARSE + j] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BS; k++) {
            for (int i = 0; i < COARSE; i++) {
                float a_val = As[ty * COARSE + i][k];
                for (int j = 0; j < COARSE; j++) {
                    float b_val = Bs[k][tx * COARSE + j];
                    accum[i][j] += a_val * b_val;
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < COARSE; i++) {
        int globalRow = row_tile + ty * COARSE + i;
        if (globalRow < M) {
            for (int j = 0; j < COARSE; j++) {
                int globalCol = col_tile + tx * COARSE + j;
                if (globalCol < N)
                    C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}










//------------------------------------------------------------------------------
// Kernel launcher function
//
// This function selects which kernel to run based on kernelId.
// Initially each kernel is implemented in a “naive” (correct but slow) way.
// Students are meant to modify/fill in the kernels and tune the launch parameters.
extern "C" void launchMatMulKernel(int kernelId, float *A, float *B, float *C, int M, int N, int K) {
    // For demonstration, we use kernelId values 1-4.
    // (Other values can be added as more kernels are implemented.)
    if (kernelId == 1) {
        // Naive kernel: one thread per output element.
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 2) {
        // Coalesced indexing version.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        matmul2_coalesced<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 3) {
        // 2x2 coarsened kernel.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        coarsened_matmul2x2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 4) {
        // Shared memory tiled version.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        MatMulTiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);

    } else if (kernelId == 5) {
        int coarsenFactor = 4; 
        dim3 blockDim(16, 16);
        dim3 gridDim((N + (blockDim.x * coarsenFactor) - 1) / (blockDim.x * coarsenFactor),
                     (M + (blockDim.y * coarsenFactor) - 1) / (blockDim.y * coarsenFactor));
        MatmulBest<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    
    else {
        // Default: if an unknown kernelId is passed, run the naive version.
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    // Make sure the kernel has finished.
    cudaDeviceSynchronize();
}
