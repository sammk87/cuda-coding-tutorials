### **Real-World CUDA Applications**

CUDA is widely used in real-world applications like image processing, numerical computations, simulations, and machine learning. Below are detailed explanations and code examples for these topics.

---

### **1. Image Processing: Grayscale Conversion with CUDA**

#### **What Does It Do?**  
This program converts a color image (RGB) to grayscale using a CUDA kernel. Each thread processes one pixel.

#### **Explanation**  
- **Grayscale Formula**: \( Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B \)
- **Thread Mapping**: Each thread computes the grayscale value for one pixel.

---

#### **Code**  
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 512
#define HEIGHT 512

// Kernel for grayscale conversion
__global__ void grayscaleKernel(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB has 3 channels
        unsigned char r = rgb[idx];
        unsigned char g = rgb[idx + 1];
        unsigned char b = rgb[idx + 2];

        gray[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

int main() {
    unsigned char *h_rgb, *h_gray; // Host memory
    unsigned char *d_rgb, *d_gray; // Device memory
    int imgSize = WIDTH * HEIGHT * 3;  // RGB image size
    int graySize = WIDTH * HEIGHT;     // Grayscale image size

    // Allocate host memory
    h_rgb = (unsigned char *)malloc(imgSize);
    h_gray = (unsigned char *)malloc(graySize);

    // Initialize image data (dummy initialization for demonstration)
    for (int i = 0; i < imgSize; i++) {
        h_rgb[i] = rand() % 256;
    }

    // Allocate device memory
    cudaMalloc(&d_rgb, imgSize);
    cudaMalloc(&d_gray, graySize);

    // Copy data to device
    cudaMemcpy(d_rgb, h_rgb, imgSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    grayscaleKernel<<<numBlocks, threadsPerBlock>>>(d_rgb, d_gray, WIDTH, HEIGHT);

    // Copy result back to host
    cudaMemcpy(h_gray, d_gray, graySize, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(h_rgb);
    free(h_gray);

    std::cout << "Grayscale conversion completed.\n";
    return 0;
}
```

---

### **2. Numerical Computation: Jacobi Iteration**

#### **What Does It Do?**  
This program solves a system of linear equations using the Jacobi iterative method in CUDA. 

#### **Explanation**  
- **Jacobi Method**: Iteratively refines solutions by solving for each variable while keeping others fixed.
- **Thread Mapping**: Each thread computes the update for one element.

---

#### **Code**
```cpp
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define N 256
#define TOLERANCE 1e-4
#define MAX_ITER 1000

// Kernel for Jacobi iteration
__global__ void jacobiKernel(float *d_x, float *d_b, float *d_A, float *d_xNew, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                sum += d_A[i * n + j] * d_x[j];
            }
        }
        d_xNew[i] = (d_b[i] - sum) / d_A[i * n + i];
    }
}

int main() {
    float *h_A, *h_b, *h_x; // Host memory
    float *d_A, *d_b, *d_x, *d_xNew; // Device memory
    size_t matrixSize = N * N * sizeof(float);
    size_t vectorSize = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(matrixSize);
    h_b = (float *)malloc(vectorSize);
    h_x = (float *)malloc(vectorSize);

    // Initialize A, b, x
    for (int i = 0; i < N; i++) {
        h_b[i] = 1.0f;
        h_x[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = (i == j) ? 2.0f : 1.0f; // Diagonal dominance
        }
    }

    // Allocate device memory
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_b, vectorSize);
    cudaMalloc(&d_x, vectorSize);
    cudaMalloc(&d_xNew, vectorSize);

    // Copy data to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, vectorSize, cudaMemcpyHostToDevice);

    // Jacobi iteration
    dim3 threadsPerBlock(128);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    float error = 1.0f;
    int iter = 0;

    while (error > TOLERANCE && iter < MAX_ITER) {
        jacobiKernel<<<numBlocks, threadsPerBlock>>>(d_x, d_b, d_A, d_xNew, N);

        // Swap x and xNew
        std::swap(d_x, d_xNew);
        iter++;

        // Compute error (on host for simplicity)
        cudaMemcpy(h_x, d_x, vectorSize, cudaMemcpyDeviceToHost);
        error = 0.0f;
        for (int i = 0; i < N; i++) {
            error += fabs(h_x[i] - h_b[i]);
        }
        error /= N;
    }

    std::cout << "Jacobi iteration converged in " << iter << " iterations.\n";

    // Free memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_xNew);
    free(h_A);
    free(h_b);
    free(h_x);

    return 0;
}
```

---

### **CUDA-Based Neural Network Example**

#### **What Does It Do?**  
This program demonstrates a simple CUDA-based feedforward neural network for a single layer with one activation function. It performs matrix-vector multiplication to compute the output of the layer.

---

### **Explanation**  
- **Neural Network Basics**: Each layer computes:  
  \[
  y = \sigma(Wx + b)
  \]
  Where:
  - \(W\) is the weight matrix.
  - \(x\) is the input vector.
  - \(b\) is the bias vector.
  - \(\sigma\) is the activation function (e.g., ReLU or sigmoid).
- **Parallelism**: Each thread computes one element of the output vector.

---

#### **Code**

```cpp
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define N_INPUT 1024  // Number of input features
#define N_OUTPUT 256  // Number of output neurons
#define THREADS_PER_BLOCK 256

// Activation function: ReLU
__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

// Kernel to compute a single forward pass of a fully connected layer
__global__ void forwardPassKernel(float *d_input, float *d_weights, float *d_bias, float *d_output, int n_input, int n_output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n_output) {
        float sum = 0.0f;
        for (int i = 0; i < n_input; i++) {
            sum += d_weights[idx * n_input + i] * d_input[i];
        }
        d_output[idx] = relu(sum + d_bias[idx]);
    }
}

int main() {
    float *h_input, *h_weights, *h_bias, *h_output;
    float *d_input, *d_weights, *d_bias, *d_output;

    // Allocate host memory
    h_input = (float *)malloc(N_INPUT * sizeof(float));
    h_weights = (float *)malloc(N_INPUT * N_OUTPUT * sizeof(float));
    h_bias = (float *)malloc(N_OUTPUT * sizeof(float));
    h_output = (float *)malloc(N_OUTPUT * sizeof(float));

    // Initialize input, weights, and biases
    for (int i = 0; i < N_INPUT; i++) {
        h_input[i] = 1.0f;  // Example: all inputs are 1.0
    }

    for (int i = 0; i < N_INPUT * N_OUTPUT; i++) {
        h_weights[i] = 0.01f;  // Example: small random weights
    }

    for (int i = 0; i < N_OUTPUT; i++) {
        h_bias[i] = 0.1f;  // Example: small biases
    }

    // Allocate device memory
    cudaMalloc(&d_input, N_INPUT * sizeof(float));
    cudaMalloc(&d_weights, N_INPUT * N_OUTPUT * sizeof(float));
    cudaMalloc(&d_bias, N_OUTPUT * sizeof(float));
    cudaMalloc(&d_output, N_OUTPUT * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, N_INPUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, N_INPUT * N_OUTPUT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, N_OUTPUT * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks((N_OUTPUT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    forwardPassKernel<<<blocks, threads>>>(d_input, d_weights, d_bias, d_output, N_INPUT, N_OUTPUT);

    // Copy results back to host
    cudaMemcpy(h_output, d_output, N_OUTPUT * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some results
    std::cout << "Output of the neural network:\n";
    for (int i = 0; i < 10; i++) { // Print first 10 outputs
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output);

    return 0;
}
```

---

### **Key Points**
1. **Input**: A single input vector processed by a fully connected layer.
2. **Weights and Biases**: Randomly initialized for simplicity.
3. **Activation Function**: ReLU implemented on the GPU.

---

### **How to Run the Code**
1. Compile the code:  
   ```bash
   nvcc -o nn_cuda nn_cuda.cu
   ```
2. Run the executable:  
   ```bash
   ./nn_cuda
   ```

---
