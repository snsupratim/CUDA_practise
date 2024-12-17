% % cuda
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10 // Number of elements in each vector

        // Kernel function to add vectors
        __global__ void
        vectorAdd(int *A, int *B, int *C, int *D, int n)
{
    int tid = threadIdx.x;

    if (tid < n)
    {
        // Perform element-wise addition and store it in vector D
        D[tid] = A[tid] + B[tid] + C[tid];
    }
}

int main()
{
    int h_A[N], h_B[N], h_C[N], h_D[N]; // Host vectors
    int *d_A, *d_B, *d_C, *d_D;         // Device vectors

    // Initialize vectors A, B, and C
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i + 1;       // Vector A: 1, 2, 3, ..., 10
        h_B[i] = (i + 1) * 2; // Vector B: 2, 4, 6, ..., 20
        h_C[i] = (i + 1) * 3; // Vector C: 3, 6, 9, ..., 30
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, N * sizeof(int));
    cudaMalloc((void **)&d_B, N * sizeof(int));
    cudaMalloc((void **)&d_C, N * sizeof(int));
    cudaMalloc((void **)&d_D, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with N threads
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, d_D, N);

    // Copy the result from device to host
    cudaMemcpy(h_D, d_D, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Resulting vector D after adding A, B, and C:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_D[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}
