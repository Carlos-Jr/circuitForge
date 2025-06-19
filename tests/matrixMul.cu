/*
 * matrixMul.cu
 *
 * Exemplo de multiplicação de matrizes em CUDA para placas com compute capability 6.1 (ex: GTX1080).
 * Compilação recomendada:
 * nvcc matrixMul.cu -O2 \
 *   -gencode arch=compute_61,code=sm_61 \
 *   -gencode arch=compute_61,code=compute_61 \
 *   -o matrixMul
 *
 * Isso gera binário (cubin) nativo para SM_61 e PTX compatível 6.1.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel de multiplicação de matrizes A e B: C = A * B
__global__ void matMul(const float* A, const float* B, float* C,
                       int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float sum = 0.0f;
        // Usar índices row-major
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    int N = 1024; // linhas de A e C
    int M = 1024; // colunas de A e linhas de B
    int K = 1024; // colunas de B e C

    size_t sizeA = N * M * sizeof(float);
    size_t sizeB = M * K * sizeof(float);
    size_t sizeC = N * K * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Inicialização simplificada
    for (int i = 0; i < N * M; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < M * K; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    matMul<<<grid, block>>>(d_A, d_B, d_C, N, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verificação simples
    printf("C[0] = %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
