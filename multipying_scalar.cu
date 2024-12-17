% % cuda
#include <stdio.h>
#include <cuda_runtime.h>

        // Kernel function to multiply three scalars
        __global__ void
        scalarMultiply(float *a, float *b, float *c, float *result)
{
    // Perform the multiplication and store the result
    *result = (*a) * (*b) * (*c);
}

int main()
{
    // Declare and initialize the scalar variables
    float h_a = 2.5f, h_b = 3.5f, h_c = 4.0f; // Host variables
    float h_result = 0.0f;                    // Host variable for storing result

    // Device variables
    float *d_a, *d_b, *d_c, *d_result;

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, sizeof(float));
    cudaMalloc((void **)&d_b, sizeof(float));
    cudaMalloc((void **)&d_c, sizeof(float));
    cudaMalloc((void **)&d_result, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel (1 block, 1 thread)
    scalarMultiply<<<1, 1>>>(d_a, d_b, d_c, d_result);

    // Copy the result from device to host
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("The result of multiplying %.2f, %.2f, and %.2f is: %.2f\n", h_a, h_b, h_c, h_result);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_result);

    return 0;
}
