### **Multi-GPU Programming in CUDA**

Multi-GPU programming allows workload distribution across multiple GPUs and enables efficient memory sharing between them. Below are examples demonstrating basic multi-GPU programming and peer-to-peer memory access.

---

### **1. Multi-GPU Basics: Distribute Workload Across GPUs**

#### **What Does It Do?**  
This program distributes the computation of squaring an array's elements across multiple GPUs. Each GPU handles a segment of the array, demonstrating workload distribution using `cudaSetDevice`.

#### **Explanation**  
- **`cudaSetDevice`**: Specifies the GPU for subsequent operations.
- **Workload Partitioning**: The array is split evenly among available GPUs.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Total number of elements
#define THREADS_PER_BLOCK 256

// Kernel to square array elements
__global__ void squareKernel(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] *= data[idx];
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cout << "This program requires at least 2 GPUs.\n";
        return 1;
    }

    // Divide workload across GPUs
    int chunkSize = N / deviceCount;
    float *h_data = (float *)malloc(N * sizeof(float));
    float *d_data[deviceCount];

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }

    // Allocate memory on each GPU and distribute data
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);  // Set the current device
        cudaMalloc(&d_data[i], chunkSize * sizeof(float));
        cudaMemcpy(d_data[i], h_data + i * chunkSize, chunkSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch kernels on each GPU
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        squareKernel<<<(chunkSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data[i], chunkSize);
    }

    // Copy results back to host
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaMemcpy(h_data + i * chunkSize, d_data[i], chunkSize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Print some results
    std::cout << "Squared data:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    for (int i = 0; i < deviceCount; i++) {
        cudaSetDevice(i);
        cudaFree(d_data[i]);
    }
    free(h_data);

    return 0;
}
```

---

### **2. Peer-to-Peer Memory Access: Sharing Data Between GPUs**

#### **What Does It Do?**  
This program demonstrates peer-to-peer memory access between two GPUs using `cudaMemcpyPeer`. One GPU computes the square of an array, and the result is transferred directly to the second GPU.

#### **Explanation**  
- **Peer Access**: Enabled with `cudaDeviceEnablePeerAccess`.
- **`cudaMemcpyPeer`**: Transfers data directly between GPUs without routing through the host.

---

#### **Code**
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 512
#define THREADS_PER_BLOCK 256

// Kernel to square array elements
__global__ void squareKernel(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] *= data[idx];
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        std::cout << "This program requires at least 2 GPUs.\n";
        return 1;
    }

    // Enable peer access
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    float *d_data1, *d_data2;
    size_t size = N * sizeof(float);

    // Allocate memory on both GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_data1, size);
    cudaSetDevice(1);
    cudaMalloc(&d_data2, size);

    // Initialize data on GPU 0
    float *h_data = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    cudaSetDevice(0);
    cudaMemcpy(d_data1, h_data, size, cudaMemcpyHostToDevice);

    // Compute square on GPU 0
    cudaSetDevice(0);
    squareKernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_data1, N);

    // Transfer result to GPU 1
    cudaMemcpyPeer(d_data2, 1, d_data1, 0, size);

    // Verify on GPU 1
    cudaSetDevice(1);
    cudaMemcpy(h_data, d_data2, size, cudaMemcpyDeviceToHost);

    std::cout << "Data on GPU 1 after transfer:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaSetDevice(0);
    cudaFree(d_data1);
    cudaSetDevice(1);
    cudaFree(d_data2);
    free(h_data);

    return 0;
}
```

---

### **Summary**

1. **Multi-GPU Basics**: Workload distribution across GPUs using `cudaSetDevice`.
2. **Peer-to-Peer Memory Access**: Enables direct data sharing between GPUs for efficient multi-GPU programming.

Use these examples as a foundation for building complex multi-GPU applications. Let me know if you need further assistance!