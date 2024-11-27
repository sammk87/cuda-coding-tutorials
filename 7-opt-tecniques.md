### **Optimization Techniques in CUDA**

CUDA optimization techniques help improve kernel performance by tuning thread configurations, unrolling loops, and minimizing warp divergence. Below are detailed explanations and code examples for these topics.

---

### **1. Kernel Performance Tuning**

#### **What Does It Do?**  
This program compares the performance of a kernel with different thread block sizes. The goal is to determine the optimal block size for a given problem.

#### **Explanation**  
- **Thread Block Size**: Affects kernel execution efficiency. Choosing the right size balances computation and resource utilization.
- **Performance Comparison**: Uses the `cudaEvent` API to measure execution time for various block sizes.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements

// Kernel to double elements
__global__ void doubleKernel(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    float *h_data, *d_data;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int blockSizes[] = {32, 64, 128, 256, 512};
    for (int blockSize : blockSizes) {
        // Create CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start event
        cudaEventRecord(start);

        // Launch kernel
        doubleKernel<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_data);

        // Record stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        std::cout << "Block size: " << blockSize << ", Execution time: " << elapsedTime << " ms\n";

        // Clean up events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

---

### **2. Loop Unrolling**

#### **What Does It Do?**  
This program compares the performance of a kernel with and without loop unrolling to illustrate its impact on performance.

#### **Explanation**  
- **Loop Unrolling**: Manually unrolls loop iterations to reduce loop control overhead and improve performance.
- **Manual Unrolling**: Unrolling a loop with a fixed number of iterations increases instruction-level parallelism.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Array size
#define THREADS_PER_BLOCK 256

// Kernel without loop unrolling
__global__ void sumKernelNoUnroll(float *data, float *result) {
    __shared__ float shared[THREADS_PER_BLOCK];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        shared[threadIdx.x] = data[idx];
    } else {
        shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[blockIdx.x] = shared[0];
    }
}

// Kernel with loop unrolling
__global__ void sumKernelUnroll(float *data, float *result) {
    __shared__ float shared[THREADS_PER_BLOCK];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        shared[threadIdx.x] = data[idx];
    } else {
        shared[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Manually unrolled loop
    if (threadIdx.x % 8 == 0) {
        shared[threadIdx.x] += shared[threadIdx.x + 1];
        shared[threadIdx.x] += shared[threadIdx.x + 2];
        shared[threadIdx.x] += shared[threadIdx.x + 3];
        shared[threadIdx.x] += shared[threadIdx.x + 4];
        shared[threadIdx.x] += shared[threadIdx.x + 5];
        shared[threadIdx.x] += shared[threadIdx.x + 6];
        shared[threadIdx.x] += shared[threadIdx.x + 7];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        result[blockIdx.x] = shared[0];
    }
}

int main() {
    float *h_data, *h_result;
    float *d_data, *d_result;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(size);
    h_result = (float *)malloc(size / THREADS_PER_BLOCK);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_result, size / THREADS_PER_BLOCK);

    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernels and compare performance
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel without unrolling
    cudaEventRecord(start);
    sumKernelNoUnroll<<<blocks, threads>>>(d_data, d_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float noUnrollTime;
    cudaEventElapsedTime(&noUnrollTime, start, stop);

    // Kernel with unrolling
    cudaEventRecord(start);
    sumKernelUnroll<<<blocks, threads>>>(d_data, d_result);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float unrollTime;
    cudaEventElapsedTime(&unrollTime, start, stop);

    std::cout << "No unrolling time: " << noUnrollTime << " ms\n";
    std::cout << "Unrolling time: " << unrollTime << " ms\n";

    // Free memory
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    free(h_result);

    return 0;
}
```

---

### **3. Warp Divergence**

#### **What Does It Do?**  
This program illustrates warp divergence when threads in the same warp follow different execution paths due to conditional statements.

#### **Explanation**  
- **Warp**: A group of 32 threads that execute in lockstep.
- **Warp Divergence**: Occurs when threads in a warp take different branches, leading to serialization.
- **Solution**: Avoid conditions that split execution paths.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define THREADS_PER_BLOCK 256

// Kernel with warp divergence
__global__ void warpDivergence(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        if (idx % 2 == 0) { // Divergent condition
            data[idx] *= 2.0f;
        } else {
            data[idx] *= 3.0f;
        }
    }
}

// Kernel avoiding warp divergence
__global__ void avoidWarpDivergence(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        float factor = (idx % 2 == 0) ? 2.0f : 3.0f; // Unified condition
        data[idx] *= factor;
    }
}

int main() {
    float *h_data, *d_data;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel with warp divergence
    warpDivergence<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data);

    // Copy back and print results
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    std::cout << "Results with warp divergence (partial):\n";
    for

 (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Reset data
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Launch kernel avoiding warp divergence
    avoidWarpDivergence<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data);

    // Copy back and print results
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    std::cout << "Results avoiding warp divergence (partial):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

---

### **Summary**
1. **Kernel Performance Tuning**: Optimize thread block sizes to maximize performance.
2. **Loop Unrolling**: Reduce loop control overhead for faster computation.
3. **Warp Divergence**: Minimize branching within warps to improve execution efficiency. 

These examples showcase practical CUDA optimization techniques. Let me know if further clarification is needed!