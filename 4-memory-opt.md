### **Memory Optimization in CUDA**

Memory optimization is crucial for achieving high performance in CUDA programs. Below are explanations and code examples for three key optimization techniques: shared memory, coalesced memory access, and unified memory.

---

### **1. Shared Memory: Optimize Matrix Multiplication**

#### **What Does It Do?**  
This program optimizes matrix multiplication using shared memory. Shared memory allows threads within a block to share data, reducing global memory accesses and improving performance.

#### **Explanation**  
- **Shared Memory**: Fast, on-chip memory shared among threads in a block.
- **Thread Collaboration**: Each thread block computes a submatrix of the result by collaborating via shared memory.
- **Blocking**: The input matrices are divided into smaller tiles that fit in shared memory.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Tile size for shared memory
#define N 256         // Matrix dimension (N x N)

// Kernel function for matrix multiplication using shared memory
__global__ void matrixMulShared(float *A, float *B, float *C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE]; // Shared memory for A
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE]; // Shared memory for B

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Global row index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Global column index
    float sum = 0.0;

    // Loop over tiles of A and B
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < n && t * TILE_SIZE + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (col < n && t * TILE_SIZE + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads(); // Synchronize threads to ensure data is loaded

        // Compute partial result for the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        __syncthreads(); // Synchronize threads before loading the next tile
    }

    // Write result to global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixMulShared<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print partial result
    std::cout << "Matrix multiplication result (partial):\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i * N] << " ";
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

### **2. Coalesced Memory Access**

#### **What Does It Do?**  
This program compares coalesced and non-coalesced memory access by reading an array and measuring performance. Coalesced access ensures adjacent threads access adjacent memory locations, reducing memory transaction overhead.

#### **Explanation**  
- **Coalesced Access**: Threads read/write consecutive memory locations.
- **Non-Coalesced Access**: Threads access scattered memory locations.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 1024  // Array size

// Kernel with coalesced memory access
__global__ void coalescedAccess(float *data, float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        result[idx] = data[idx] * 2.0f; // Coalesced access
    }
}

// Kernel with non-coalesced memory access
__global__ void nonCoalescedAccess(float *data, float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        result[idx] = data[(N - 1) - idx] * 2.0f; // Non-coalesced access
    }
}

int main() {
    float *h_data, *h_result;
    float *d_data, *d_result;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(size);
    h_result = (float *)malloc(size);

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_result, size);

    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Measure coalesced access
    auto start = std::chrono::high_resolution_clock::now();
    coalescedAccess<<<(N + 255) / 256, 256>>>(d_data, d_result);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Coalesced access time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    // Measure non-coalesced access
    start = std::chrono::high_resolution_clock::now();
    nonCoalescedAccess<<<(N + 255) / 256, 256>>>(d_data, d_result);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Non-coalesced access time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    // Free memory
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);

    return 0;
}
```

---

### **3. Unified Memory**

#### **What Does It Do?**  
This program demonstrates unified memory, where both the host and device share a single memory space. This simplifies memory management as explicit memory copying is unnecessary.

#### **Explanation**  
- **Unified Memory**: Allocated using `cudaMallocManaged`. Accessible by both CPU and GPU.
- **Simplified Access**: No need for `cudaMemcpy` to transfer data.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Array size

// Kernel to double elements in unified memory
__global__ void doubleElements(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    float *data;

    // Allocate unified memory
    cudaMallocManaged(&data, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }

    // Launch kernel
    doubleElements<<<(N + 255) / 256, 256>>>(data);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Print some results
    std::cout << "Doubled elements:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
