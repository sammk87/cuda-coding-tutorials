### **Using Python for CUDA Programming**

Python is a powerful and user-friendly programming language, and when paired with **PyCUDA** or **CuPy**, it enables developers to write CUDA programs with ease. These libraries provide Python bindings to NVIDIA's CUDA framework, allowing seamless GPU computation.

---

### **Key Advantages of Using Python for CUDA Programming**

1. **Ease of Use**:
   - Python’s high-level syntax makes GPU programming more accessible compared to C++.
   - Simplifies memory management and kernel execution with clean APIs.

2. **Rapid Development**:
   - Reduces boilerplate code, allowing developers to focus on algorithm development.
   - Quicker prototyping of GPU-based algorithms.

3. **Integration**:
   - Integrates easily with Python’s scientific libraries like NumPy, SciPy, and Matplotlib.
   - Supports workflows that combine CPU and GPU computations seamlessly.

4. **Dynamic Compilation**:
   - PyCUDA compiles CUDA kernels at runtime, enabling on-the-fly modifications.

5. **Visualization**:
   - Python’s rich ecosystem (e.g., Matplotlib, Seaborn) can be used to visualize GPU computation results easily.

---

### **Set of Examples: Python for CUDA Programming**

#### **1. Basic CUDA Kernel Execution**

This example demonstrates a basic CUDA kernel for squaring numbers in an array.

```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for squaring numbers
kernel_code = """
__global__ void square(float *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] *= arr[idx];
    }
}
"""

# Compile the CUDA kernel
mod = SourceModule(kernel_code)
square_kernel = mod.get_function("square")

# Input data
n = 1024
arr = np.linspace(1, 32, n).astype(np.float32)

# Allocate device memory
d_arr = cuda.mem_alloc(arr.nbytes)

# Copy data to the device
cuda.memcpy_htod(d_arr, arr)

# Launch the kernel
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
square_kernel(d_arr, np.int32(n), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

# Copy results back to host
cuda.memcpy_dtoh(arr, d_arr)

# Print results
print("Squared Array (first 10 elements):", arr[:10])
```

---

#### **2. Memory Management with Unified Memory**

Unified memory simplifies memory management by automatically handling data movement between the CPU and GPU.

```python
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Input size
n = 1024

# Allocate unified memory
arr = cuda.managed_zeros(n, dtype=np.float32)
arr[:] = np.linspace(1, 32, n).astype(np.float32)

# CUDA kernel for squaring numbers
kernel_code = """
__global__ void squareUnified(float *arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] *= arr[idx];
    }
}
"""

# Compile and launch the kernel
mod = SourceModule(kernel_code)
square_kernel = mod.get_function("squareUnified")

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
square_kernel(arr, np.int32(n), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

# Synchronize to ensure the kernel is finished
cuda.Context.synchronize()

# Print results
print("Unified Memory Squared Array (first 10 elements):", arr[:10])
```

---

#### **3. Matrix Multiplication with PyCUDA**

Demonstrates how to perform matrix multiplication using a custom CUDA kernel.

```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for matrix multiplication
kernel_code = """
__global__ void matMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
matmul_kernel = mod.get_function("matMul")

# Matrix dimensions
N = 32
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Allocate device memory
d_A = cuda.mem_alloc(A.nbytes)
d_B = cuda.mem_alloc(B.nbytes)
d_C = cuda.mem_alloc(C.nbytes)

# Copy data to the device
cuda.memcpy_htod(d_A, A)
cuda.memcpy_htod(d_B, B)

# Launch the kernel
threads_per_block = (16, 16, 1)
blocks_per_grid = (N // 16, N // 16, 1)
matmul_kernel(d_A, d_B, d_C, np.int32(N), block=threads_per_block, grid=blocks_per_grid)

# Copy result back to host
cuda.memcpy_dtoh(C, d_C)

# Print result
print("Matrix Multiplication Result (first 5x5 block):\n", C[:5, :5])
```

---

#### **4. Benchmarking CUDA Kernels**

Measure execution time of a kernel using Python’s `time.perf_counter`.

```python
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

# CUDA kernel for adding vectors
kernel_code = """
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
add_kernel = mod.get_function("add")

# Vector size
n = 1024 * 1024
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)
c = np.zeros_like(a)

# Allocate device memory
d_a = cuda.mem_alloc(a.nbytes)
d_b = cuda.mem_alloc(b.nbytes)
d_c = cuda.mem_alloc(c.nbytes)

# Copy data to device
cuda.memcpy_htod(d_a, a)
cuda.memcpy_htod(d_b, b)

# Measure execution time
start = time.perf_counter()
add_kernel(d_a, d_b, d_c, np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1))
cuda.Context.synchronize()
end = time.perf_counter()

# Copy result back to host
cuda.memcpy_dtoh(c, d_c)

print(f"Kernel execution time: {end - start:.6f} seconds")
```

---

### **Conclusion**

**Advantages of Python for CUDA Programming**:
1. High-level syntax makes GPU programming accessible.
2. Seamless integration with Python's scientific stack.
3. Easier debugging and visualization of results.
4. Rapid prototyping for computational tasks.

These examples demonstrate Python's versatility for CUDA programming, from basic kernels to complex operations like matrix multiplication and performance benchmarking. Let me know if you need further assistance!