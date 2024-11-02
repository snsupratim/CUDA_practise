#include <stdio.h>
#include <cuda.h>

#define N 10 // Number of Fibonacci numbers to generate

// CUDA kernel to calculate Fibonacci numbers
__global__ void fibonacciKernel(int *fib)
{
    int idx = threadIdx.x;

    if (idx == 0)
    {
        fib[idx] = 0; // First Fibonacci number
    }
    else if (idx == 1)
    {
        fib[idx] = 1; // Second Fibonacci number
    }
    else if (idx < N)
    {
        // Calculate Fibonacci using previously computed values
        fib[idx] = fib[idx - 1] + fib[idx - 2];
    }
}

int main()
{
    int *d_fib;   // Device pointer for Fibonacci numbers
    int h_fib[N]; // Host array to store results

    // Allocate memory on the device
    cudaMalloc((void **)&d_fib, N * sizeof(int));

    // Launch the kernel with one block and N threads
    fibonacciKernel<<<1, N>>>(d_fib);

    // Copy the result from device to host
    cudaMemcpy(h_fib, d_fib, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the Fibonacci series
    printf("Fibonacci Series:\\n");
    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_fib[i]);
    }
    printf("\\n");

    // Free device memory
    cudaFree(d_fib);

    return 0;
}