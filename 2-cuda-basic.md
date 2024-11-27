### **CUDA Basics**

CUDA basics involve understanding threads and blocks, managing memory between the host and device, and learning synchronization within kernels. Below are detailed explanations and code examples for these topics.

---

### **1. CUDA Threads and Blocks**  

#### **What Does It Do?**  
This program demonstrates how to launch a CUDA kernel where each thread computes the square of an element in an array. The concept of threads and blocks is introduced to divide the workload across multiple threads.

#### **Explanation**  
- **Thread Indexing**: Each thread computes one element of the array. `threadIdx.x` and `blockIdx.x` are used to calculate the global thread ID.
- **Grid and Block**: Threads are organized in blocks, and blocks are part of a grid. The total thread ID is determined by the formula:  
  \[
  \text{globalIdx} = \text{threadIdx.x} + \text{blockIdx.x} \times \text{blockDim.x}
  \]

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Array size
#define THREADS_PER_BLOCK 256

// Kernel to compute the square of each element
__global__ void squareKernel(float *d_array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    if (idx < N) {
        d_array[idx] = d_array[idx] * d_array[idx];
    }
}

int main() {
    float *h_array, *d_array; // Host and device arrays
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_array = (float *)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, size);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Launch kernel with (N / THREADS_PER_BLOCK) blocks and THREADS_PER_BLOCK threads per block
    squareKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_array);

    // Copy results back to host
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Print some results
    for (int i = 0; i < 10; i++) {
        std::cout << "h_array[" << i << "] = " << h_array[i] << "\n";
    }

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
```

---

### **2. Memory Management**

#### **What Does It Do?**  
This program shows how to allocate memory on the GPU using `cudaMalloc` and transfer data between the CPU (host) and GPU (device) using `cudaMemcpy`.

#### **Explanation**  
- **Memory Allocation**: `cudaMalloc` allocates memory on the GPU, and `cudaFree` releases it.
- **Memory Transfer**: `cudaMemcpy` is used to copy data between host and device memory. Transfer directions include:
  - `cudaMemcpyHostToDevice`
  - `cudaMemcpyDeviceToHost`

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 10

int main() {
    int *h_array, *d_array; // Host and device arrays
    size_t size = N * sizeof(int);

    // Allocate memory on host
    h_array = (int *)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = i;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_array, size);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Modify data on host for verification
    for (int i = 0; i < N; i++) {
        h_array[i] = 0;
    }

    // Copy data back from device to host
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Data after copying back from device:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
```

---

### **3. Synchronization**

#### **What Does It Do?**  
This program demonstrates the use of `__syncthreads()` to synchronize threads within a block. Synchronization ensures that all threads in a block complete a specific part of the computation before moving forward.

#### **Explanation**  
- **`__syncthreads()`**: Synchronizes all threads in a block. This is important when threads depend on the results of others (e.g., shared memory calculations).
- **Shared Memory**: Memory shared by threads in the same block, used for collaborative computations.

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 256  // Array size (must match THREADS_PER_BLOCK)
#define THREADS_PER_BLOCK 256

// Kernel to compute prefix sum using synchronization
__global__ void prefixSumKernel(int *d_array) {
    __shared__ int temp[THREADS_PER_BLOCK]; // Shared memory
    int idx = threadIdx.x;

    // Load data into shared memory
    temp[idx] = d_array[idx];
    __syncthreads();

    // Perform prefix sum (inclusive)
    for (int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        int val = 0;
        if (idx >= stride) {
            val = temp[idx - stride];
        }
        __syncthreads();
        temp[idx] += val;
        __syncthreads();
    }

    // Write results back to global memory
    d_array[idx] = temp[idx];
}

int main() {
    int *h_array, *d_array; // Host and device arrays
    size_t size = N * sizeof(int);

    // Allocate and initialize host memory
    h_array = (int *)malloc(size);
    for (int i = 0; i < N; i++) {
        h_array[i] = 1; // Initialize with ones
    }

    // Allocate device memory
    cudaMalloc((void **)&d_array, size);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block of THREADS_PER_BLOCK threads
    prefixSumKernel<<<1, THREADS_PER_BLOCK>>>(d_array);

    // Copy results back to host
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Prefix sum result:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
```

---

### **Summary**
1. **Threads and Blocks**: Uses thread indexing to perform operations on arrays in parallel.
2. **Memory Management**: Demonstrates allocation and data transfer between host and device.
3. **Synchronization**: Shows how threads collaborate using shared memory and synchronization.

Let me know if you'd like further details or additional examples!