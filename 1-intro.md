### **Introduction to CUDA**

CUDA (Compute Unified Device Architecture) is NVIDIA’s platform for parallel computing, allowing developers to leverage the massive parallelism of GPUs to accelerate computations. Below, we break down three essential parts of a CUDA introduction, explaining their purpose and functionality.

---

### **1.1 Understanding GPU vs. CPU**

#### **What Does It Do?**

This program compares the execution time of a simple addition operation on the CPU and GPU. It highlights the GPU's advantage in parallel processing when working with large datasets.

#### **Explanation**

- **Host and Device**: The CPU is referred to as the "host," while the GPU is the "device."
- **Memory Management**: Data is copied from host memory to device memory for GPU computation and back to the host to retrieve results.
- **Parallelism**: The GPU kernel launches many threads to perform operations simultaneously, whereas the CPU performs them sequentially.

---

#### **Code**

```cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define N 1000000  // Number of elements in the arrays

// GPU kernel for element-wise addition
__global__ void add_GPU(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the global thread index
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU function for element-wise addition
void add_CPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *a, *b, *c; // Host arrays
    int *d_a, *d_b, *d_c; // Device arrays
    int size = N * sizeof(int);

    // Allocate memory for host arrays
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize arrays with values
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Measure time for CPU computation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    add_CPU(a, b, c, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count() 
              << " us\n";

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Measure time for GPU computation
    auto start_gpu = std::chrono::high_resolution_clock::now();
    add_GPU<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize(); // Ensure GPU computation is complete
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "GPU time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count() 
              << " us\n";

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
```

---

### **1.2 Basic CUDA Program Structure**

#### **What Does It Do?**

This program is a "Hello, World!" example for CUDA, showcasing how to write and launch a simple kernel on the GPU.

#### **Explanation**

- **Kernel**: A kernel is a function executed on the GPU. It’s written using the `__global__` keyword.
- **Thread Management**: Each thread in a block executes the kernel function. In this example, we launch one thread in one block.
- **Synchronization**: `cudaDeviceSynchronize()` ensures that the host waits for the GPU to finish execution.

---

#### **Code**

```cpp
#include <iostream>
#include <cuda_runtime.h>

// A simple kernel that prints from the GPU
__global__ void helloCUDA() {
    printf("Hello, World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 1 thread
    helloCUDA<<<1, 1>>>();

    // Wait for the GPU to finish execution
    cudaDeviceSynchronize();

    return 0;
}
```

---

### **1.3 Setting Up the CUDA Environment**

#### **What Does It Do?**

This program checks the system for CUDA-capable devices and prints their properties.

#### **Explanation**

- **Device Count**: `cudaGetDeviceCount()` retrieves the number of GPUs available.
- **Device Properties**: `cudaGetDeviceProperties()` fetches detailed information about each GPU, such as its name, compute capability, memory size, and thread capacities.

---

#### **Code**

```cpp
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA-capable devices: " << deviceCount << "\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Maximum threads per block: " << prop.maxThreadsPerBlock << "\n";
    }

    return 0;
}
```

---

### **Summary**

1. **GPU vs. CPU Comparison**: Showcases the speedup provided by GPUs for parallel computations.
2. **Hello CUDA**: Introduces the basic structure of a CUDA program and how to launch kernels.
3. **Environment Setup**: Teaches how to query and understand GPU properties.

These foundational programs prepare you to explore CUDA programming and its powerful capabilities. Let me know if you want more explanations or advanced topics!