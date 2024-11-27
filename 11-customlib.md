### **Writing Custom CUDA Libraries**

Custom CUDA libraries allow you to create reusable and modular code for GPU computations. Below are examples to create a library function for vector addition and debug it using `cuda-memcheck`.

---

### **1. Creating a Reusable CUDA Function**

#### **What Does It Do?**  
This example demonstrates how to write a reusable CUDA library function for vector addition. The library is modular, enabling the inclusion of the function in different projects.

#### **Structure**  
1. **Header File (`vector_add.h`)**: Declares the kernel and helper functions.
2. **Source File (`vector_add.cu`)**: Implements the kernel and utility functions.
3. **Main File (`main.cpp`)**: Uses the library for vector addition.

---

#### **Code**  

**1. `vector_add.h` (Header File)**  
```cpp
#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>

// Function to launch the vector addition kernel
void vectorAdd(float *h_a, float *h_b, float *h_c, int n);

#endif
```

---

**2. `vector_add.cu` (Source File)**  
```cpp
#include "vector_add.h"
#include <iostream>

// Kernel for vector addition
__global__ void vectorAddKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Function to launch the vector addition kernel
void vectorAdd(float *h_a, float *h_b, float *h_c, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy results back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

---

**3. `main.cpp` (Main File)**  
```cpp
#include "vector_add.h"
#include <iostream>

#define N 1024  // Number of elements

int main() {
    float *h_a, *h_b, *h_c;

    // Allocate host memory
    h_a = (float *)malloc(N * sizeof(float));
    h_b = (float *)malloc(N * sizeof(float));
    h_c = (float *)malloc(N * sizeof(float));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Call the library function
    vectorAdd(h_a, h_b, h_c, N);

    // Print some results
    std::cout << "Vector addition results:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << "\n";
    }

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

---

#### **How to Compile and Run**  
1. Compile the library and main program:  
   ```bash
   nvcc -c vector_add.cu -o vector_add.o
   nvcc main.cpp vector_add.o -o vector_add
   ```
2. Run the executable:  
   ```bash
   ./vector_add
   ```

---

### **2. Debugging Custom Libraries**

#### **What Does It Do?**  
This demonstrates how to debug the `vectorAddKernel` using `cuda-memcheck`. It introduces a deliberate bug (e.g., an out-of-bounds memory access) and uses `cuda-memcheck` to detect it.

---

#### **Modify the Kernel to Introduce a Bug**  
Edit `vector_add.cu` as follows:  
```cpp
// Kernel with a bug: Out-of-bounds access
__global__ void vectorAddKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= n) {  // Bug: Should be idx < n, allowing out-of-bounds access
        c[idx] = a[idx] + b[idx];
    }
}
```

---

#### **Run `cuda-memcheck`**
1. Compile the code as usual.  
2. Run the program with `cuda-memcheck`:  
   ```bash
   cuda-memcheck ./vector_add
   ```

#### **Output**  
`cuda-memcheck` will detect and report the out-of-bounds access error, including the kernel name and memory location.

---

### **Summary**

1. **Reusable CUDA Function**: Encapsulated the vector addition kernel and memory management in a reusable library.
2. **Debugging with `cuda-memcheck`**: Identified out-of-bounds errors in the kernel.  

Let me know if you'd like more debugging techniques or additional examples!