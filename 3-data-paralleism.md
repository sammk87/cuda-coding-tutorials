### **Data Parallelism**

Data parallelism involves dividing data across multiple threads for simultaneous computation, enabling efficient processing of large datasets. Below are detailed explanations and code examples for key concepts in CUDA data parallelism, with clear comments for better understanding.

---

### **1. Vector Addition**

#### **What Does It Do?**  
This program performs element-wise addition of two arrays (`A` and `B`) in parallel using CUDA threads. Each thread calculates the sum for a single element, making it an excellent example of data parallelism.

#### **Explanation**  
- **Thread Indexing**: Each thread computes the result for one element in the array. The global thread ID determines which element to compute.
- **Grid and Block**: The workload is distributed across blocks of threads.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements in the arrays

// Kernel function to add two vectors
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread ID
    if (idx < n) {
        C[idx] = A[idx] + B[idx]; // Perform addition
    }
}

int main() {
    float *h_A, *h_B, *h_C;  // Host arrays
    float *d_A, *d_B, *d_C;  // Device arrays
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel with (N / 256) blocks and 256 threads per block
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some results
    std::cout << "Vector addition results:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << "\n";
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

---

### **2. Matrix Multiplication**

#### **What Does It Do?**  
This program performs basic matrix multiplication in CUDA. Each thread computes one element of the output matrix.

#### **Explanation**  
- **Matrix Representation**: Matrices are stored as 1D arrays in row-major order.
- **Thread Responsibility**: Each thread calculates the value of one element in the result matrix.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Dimensions of the matrix (N x N)

// Kernel function for matrix multiplication
__global__ void matrixMul(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < n && col < n) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col]; // Dot product
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // Launch the kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some results
    std::cout << "Matrix multiplication results (partial):\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i * N] << " "; // Print the first column
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

---

### **3. Parallel Reduction**

#### **What Does It Do?**  
This program sums the elements of an array using a reduction kernel. Threads collaboratively reduce the array size until the final result is obtained.

#### **Explanation**  
- **Reduction Steps**: Each thread computes partial sums, and `__syncthreads()` ensures synchronization at each step.
- **Shared Memory**: Threads use shared memory to hold partial sums for efficient access.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Array size
#define THREADS_PER_BLOCK 256

// Kernel function for parallel reduction
__global__ void reduceSum(float *d_array, float *d_result) {
    __shared__ float temp[THREADS_PER_BLOCK]; // Shared memory for partial sums

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    temp[tid] = (idx < N) ? d_array[idx] : 0.0f;
    __syncthreads();

    // Perform reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Write result of block to global memory
    if (tid == 0) {
        d_result[blockIdx.x] = temp[0];
    }
}

int main() {
    float *h_array, *h_result;
    float *d_array, *d_result;
    size_t size = N * sizeof(float);
    size_t result_size = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK * sizeof(float);

    // Allocate host memory
    h_array = (float *)malloc(size);
    h_result = (float *)malloc(result_size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = 1.0f; // Initialize with ones
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, size);
    cudaMalloc((void **)&d_result, result_size);

    // Copy data to device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Launch kernel
    reduceSum<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_array, d_result);

    // Copy partial results back to host
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);

    // Compute final sum on host
    float final_sum = 0.0f;
    for (int i = 0; i < (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; i++) {
        final_sum += h_result[i];
    }

    std::cout << "Sum of array elements: " << final_sum << "\n";

    // Free memory
    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);
    free