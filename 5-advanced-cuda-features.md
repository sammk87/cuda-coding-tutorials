### **Advanced CUDA Features**

Advanced CUDA features like streams, atomic operations, and dynamic parallelism enable developers to utilize GPU resources efficiently for complex problems. Below are explanations and code examples for these features.

---

### **1. Streams and Concurrency**

#### **What Does It Do?**  
This program uses CUDA streams to overlap memory transfers and kernel execution. Streams allow for asynchronous execution, improving performance by enabling the GPU to perform multiple tasks concurrently.

#### **Explanation**  
- **CUDA Streams**: Independent sequences of operations that can run concurrently.
- **Asynchronous Transfers**: `cudaMemcpyAsync` enables non-blocking data transfers.
- **Overlap**: By assigning operations to different streams, the GPU can process kernels while transferring data.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define THREADS_PER_BLOCK 256

// Kernel to double array elements
__global__ void doubleElements(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

int main() {
    float *h_data, *d_data1, *d_data2;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_data1, size);
    cudaMalloc((void **)&d_data2, size);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Asynchronous memory copy and kernel launch in stream1
    cudaMemcpyAsync(d_data1, h_data, size, cudaMemcpyHostToDevice, stream1);
    doubleElements<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream1>>>(d_data1, N);
    cudaMemcpyAsync(h_data, d_data1, size, cudaMemcpyDeviceToHost, stream1);

    // Asynchronous memory copy and kernel launch in stream2
    cudaMemcpyAsync(d_data2, h_data, size, cudaMemcpyHostToDevice, stream2);
    doubleElements<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream2>>>(d_data2, N);
    cudaMemcpyAsync(h_data, d_data2, size, cudaMemcpyDeviceToHost, stream2);

    // Wait for all streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print some results
    std::cout << "Processed data:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Clean up
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(h_data);

    return 0;
}
```

---

### **2. Atomic Operations**

#### **What Does It Do?**  
This program uses atomic operations to compute a histogram on the GPU. Atomic operations ensure that updates to shared variables are performed without race conditions.

#### **Explanation**  
- **Atomic Addition**: Ensures correct updates when multiple threads modify the same memory location.
- **Histogram**: Counts occurrences of values in an array.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define NUM_BINS 16

// Kernel to compute histogram
__global__ void computeHistogram(int *data, int *histogram, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int bin = data[idx] % NUM_BINS; // Map value to a bin
        atomicAdd(&histogram[bin], 1);  // Increment bin count atomically
    }
}

int main() {
    int *h_data, *h_histogram;
    int *d_data, *d_histogram;
    size_t dataSize = N * sizeof(int);
    size_t histogramSize = NUM_BINS * sizeof(int);

    // Allocate host memory
    h_data = (int *)malloc(dataSize);
    h_histogram = (int *)calloc(NUM_BINS, sizeof(int));

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i % 32; // Values from 0 to 31
    }

    // Allocate device memory
    cudaMalloc((void **)&d_data, dataSize);
    cudaMalloc((void **)&d_histogram, histogramSize);

    // Copy data to device
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, h_histogram, histogramSize, cudaMemcpyHostToDevice);

    // Launch kernel
    computeHistogram<<<(N + 255) / 256, 256>>>(d_data, d_histogram, N);

    // Copy result back to host
    cudaMemcpy(h_histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost);

    // Print histogram
    std::cout << "Histogram:\n";
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bin " << i << ": " << h_histogram[i] << "\n";
    }

    // Clean up
    cudaFree(d_data);
    cudaFree(d_histogram);
    free(h_data);
    free(h_histogram);

    return 0;
}
```

---

### **3. Dynamic Parallelism**

#### **What Does It Do?**  
This program demonstrates dynamic parallelism by launching a kernel from another kernel. The child kernel performs additional computations on the data.

#### **Explanation**  
- **Dynamic Parallelism**: Enables a GPU kernel to launch other kernels.
- **Parent and Child Kernels**: The parent kernel computes an initial result and launches a child kernel for further computation.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

// Child kernel to double the values
__global__ void doubleKernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// Parent kernel to launch child kernel
__global__ void parentKernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] += 1.0f; // Initial computation

        // Launch child kernel
        if (idx == 0) {
            doubleKernel<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(data, n);
            cudaDeviceSynchronize(); // Wait for child kernel to complete
        }
    }
}

int main() {
    float *data;
    size_t size = N * sizeof(float);

    // Allocate unified memory
    cudaMallocManaged(&data, size);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }

    // Launch parent kernel
    parentKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(data, N);

    // Synchronize
    cudaDeviceSynchronize();

    // Print some results
    std::cout << "Final data:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(data);

    return 0;
}
```

---

### **Summary**
1. **Streams and Concurrency**: Demonstrates overlapping data transfers and kernel execution using CUDA streams.
2. **Atomic Operations**: Ensures race-free updates for shared variables, useful in problems like histograms.
3. **Dynamic Parallelism**: Allows GPU kernels to launch other kernels, enabling hierarchical computations.

These techniques unlock advanced CUDA capabilities and help tackle complex workloads. Let me know if you need further assistance!