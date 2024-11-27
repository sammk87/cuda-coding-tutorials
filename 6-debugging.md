### **Debugging and Profiling in CUDA**

Debugging and profiling are essential skills for identifying and resolving issues in CUDA programs while optimizing performance. Below are explanations and code examples for debugging with `cuda-memcheck` and profiling with the `cudaEvent` API.

---

### **1. Debugging with `cuda-memcheck`**

#### **What Does It Do?**  
This program introduces intentional memory errors (out-of-bounds access) to demonstrate debugging with `cuda-memcheck`. Using `cuda-memcheck`, you can detect issues such as illegal memory accesses, race conditions, and invalid memory operations.

#### **Explanation**  
- **Out-of-Bounds Access**: Accessing array elements beyond allocated memory causes undefined behavior.
- **`cuda-memcheck`**: A tool that detects and reports memory errors in CUDA programs.

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 256  // Array size
#define THREADS_PER_BLOCK 128

// Kernel with an intentional out-of-bounds error
__global__ void faultyKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Intentionally access out-of-bounds memory
    if (idx <= N) { // Error: The condition allows idx == N, which is out-of-bounds
        data[idx] = idx * 2;
    }
}

int main() {
    int *h_data, *d_data;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_data = (int *)malloc(size);

    // Allocate device memory
    cudaMalloc((void **)&d_data, size);

    // Launch the kernel with an intentional error
    faultyKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data);

    // Copy data back to host (this may fail due to the error)
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

---

#### **Steps to Debug**  
1. Compile the code with debugging flags:  
   ```bash
   nvcc -g -G -o debug_faulty_kernel faulty_kernel.cu
   ```
2. Run the program with `cuda-memcheck`:  
   ```bash
   cuda-memcheck ./debug_faulty_kernel
   ```
3. Review the output to identify the memory error and its location.

---

### **2. Profiling GPU Code**

#### **What Does It Do?**  
This program measures the execution time of a kernel using the `cudaEvent` API. Profiling helps identify performance bottlenecks and optimize the code.

#### **Explanation**  
- **`cudaEvent` API**: Provides high-resolution timers for GPU events.
  - `cudaEventCreate`: Creates event objects.
  - `cudaEventRecord`: Records the event.
  - `cudaEventElapsedTime`: Measures elapsed time between two events.
- **Kernel Timing**: Measures only the kernel execution time, excluding memory transfers.

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Array size
#define THREADS_PER_BLOCK 256

// Kernel to double array elements
__global__ void doubleKernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
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
    cudaMalloc((void **)&d_data, size);

    // Copy data to device
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    doubleKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data, N);

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print elapsed time
    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

---

#### **Steps to Profile**  
1. Compile the code with `nvcc`:  
   ```bash
   nvcc -o profile_kernel profile_kernel.cu
   ```
2. Run the program:  
   ```bash
   ./profile_kernel
   ```
3. Observe the output for the kernel execution time.

---

### **Summary**  
1. **Debugging with `cuda-memcheck`**: Detects memory errors such as out-of-bounds accesses, helping debug CUDA programs.
2. **Profiling with `cudaEvent` API**: Measures kernel execution time, allowing developers to optimize performance.

These tools are essential for developing efficient and error-free CUDA applications. Let me know if you need further assistance or want additional examples!