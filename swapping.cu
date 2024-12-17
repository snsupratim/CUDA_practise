% % cuda
#include <stdio.h>
#include <cuda_runtime.h>

        // Kernel function to swap two elements without using a third variable
        __global__ void
        swapKernel(int *a, int *b)
{
    // Swap the elements using arithmetic operations (addition and subtraction)
    *a = *a + *b; // a = a + b
    *b = *a - *b; // b = (a + b) - b = a
    *a = *a - *b; // a = (a + b) - a = b
}

int main()
{
    int h_a = 5, h_b = 10; // Host variables
    int *d_a, *d_b;        // Device variables

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel (1 block, 1 thread)
    swapKernel<<<1, 1>>>(d_a, d_b);

    // Copy the result back from device to host
    cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the swapped values
    printf("After swapping, a = %d and b = %d\n", h_a, h_b);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
