% % cuda
#include <stdio.h>
#include <cuda_runtime.h>

        __global__ void
        sumKernel(int *array, int *result, int n)
{
    __shared__ int partialSum[10]; // Shared memory for partial sums

    int tid = threadIdx.x;
    partialSum[tid] = (tid < n) ? array[tid] : 0; // Load elements into shared memory

    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    // First thread in the block writes the result
    if (tid == 0)
    {
        *result = partialSum[0];
    }
}

int main()
{
    const int n = 10;
    int h_array[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // Example array of 10 numbers
    int h_result = 0;                                 // Host variable for result

    // Device variables
    int *d_array, *d_result;
    cudaMalloc((void **)&d_array, n * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));

    // Copy array from host to device
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 10 threads in a single block
    sumKernel<<<1, n>>>(d_array, d_result, n);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Sum of array elements: %d\n", h_result);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_result);

    return 0;
}
