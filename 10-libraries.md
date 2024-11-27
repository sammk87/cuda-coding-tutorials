### **CUDA Interoperability and Libraries**

CUDA interoperability with OpenGL and libraries like cuBLAS and cuFFT demonstrates how CUDA integrates with graphics and numerical libraries to accelerate computations and visualizations.

---

### **1. CUDA and OpenGL Interoperability**

#### **What Does It Do?**  
This program renders an animated 3D surface using CUDA to compute vertex positions and OpenGL for rendering. CUDA-OpenGL interop ensures seamless data sharing between CUDA kernels and OpenGL buffers.

#### **Explanation**  
- **Vertex Buffer Object (VBO)**: Used by OpenGL to store vertex data.
- **CUDA-OpenGL Interop**: Maps the VBO to CUDA, allowing CUDA kernels to manipulate vertex data directly.

---

#### **Code**  
```cpp
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cmath>
#include <iostream>

#define WIDTH 512
#define HEIGHT 512

GLuint vbo;                            // OpenGL Vertex Buffer Object
struct cudaGraphicsResource *cudaVBO;  // CUDA-OpenGL interop resource

// CUDA kernel to update vertex positions
__global__ void updateVertices(float4 *vertices, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float u = x / (float)width;
        float v = y / (float)height;
        u = u * 2.0f - 1.0f;
        v = v * 2.0f - 1.0f;

        float w = sinf(u * 10.0f + time) * cosf(v * 10.0f + time) * 0.1f;
        vertices[idx] = make_float4(u, w, v, 1.0f);
    }
}

// OpenGL display function
void display() {
    static float time = 0.0f;
    time += 0.01f;

    // Map CUDA resource
    float4 *d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cudaVBO);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vertices, &num_bytes, cudaVBO);

    // Launch CUDA kernel
    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);
    updateVertices<<<blocks, threads>>>(d_vertices, WIDTH, HEIGHT, time);

    // Unmap CUDA resource
    cudaGraphicsUnmapResources(1, &cudaVBO);

    // Render the VBO
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glDrawArrays(GL_POINTS, 0, WIDTH * HEIGHT);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

// Initialize OpenGL
void initGL() {
    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0f);
}

// Initialize CUDA-OpenGL interop
void initInterop() {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(float4), 0, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Clean up resources
void cleanup() {
    cudaGraphicsUnregisterResource(cudaVBO);
    glDeleteBuffers(1, &vbo);
}

int main(int argc, char **argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("CUDA-OpenGL Interop");

    // Initialize OpenGL and CUDA
    initGL();
    initInterop();

    // Register display callback
    glutDisplayFunc(display);
    glutCloseFunc(cleanup);

    // Start rendering
    glutMainLoop();
    return 0;
}
```

---

### **2. Using CUDA Libraries**

#### **Matrix Multiplication with cuBLAS**

#### **What Does It Do?**  
This program uses cuBLAS to perform matrix multiplication \( C = AB \).

#### **Explanation**  
- **cuBLAS**: NVIDIA's library for BLAS (Basic Linear Algebra Subprograms).
- **`cublasSgemm`**: Performs single-precision matrix multiplication.

---

#### **Code**  
```cpp
#include <iostream>
#include <cublas_v2.h>

#define N 512

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t stat, const char *msg) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": CUBLAS error\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices
    size_t size = N * N * sizeof(float);

    // Allocate and initialize host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; // Example values
        h_B[i] = 1.0f;
    }

    // Allocate device memory
    checkCudaError(cudaMalloc(&d_A, size), "Failed to allocate d_A");
    checkCudaError(cudaMalloc(&d_B, size), "Failed to allocate d_B");
    checkCudaError(cudaMalloc(&d_C, size), "Failed to allocate d_C");

    // Copy matrices to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Failed to copy h_B to d_B");

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Perform matrix multiplication
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N),
                     "Failed to perform SGEMM");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "Failed to copy d_C to h_C");

    // Print some results
    std::cout << "Result matrix:\n";
    for (int i = 0; i < 10; i++) { // Print first row
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";

    // Clean up
    cublasDestroy(handle);
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

### **Fast Fourier Transform (FFT) with cuFFT**

#### **What Does It Do?**  
This program demonstrates how to perform a 1D Fast Fourier Transform (FFT) and its inverse (IFFT) using the cuFFT library. The FFT is commonly used for signal processing, image analysis, and other computational tasks.

---

### **Explanation**  
- **cuFFT**: NVIDIA's library for efficient Fourier transforms on the GPU.
- **Steps**:
  1. Initialize input data on the host and copy it to the GPU.
  2. Create a cuFFT plan specifying the size and type of the FFT.
  3. Execute the FFT using `cufftExecC2C` for complex-to-complex transforms.
  4. Execute the inverse FFT to retrieve the original signal.
- **Scaling**: After an inverse FFT, results must be scaled by the transform size to recover the original signal.

---

### **Code**
```cpp
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>

#define N 1024  // Size of the FFT (number of points)

// Check CUDA error
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

// Check cuFFT error
void checkCufftError(cufftResult err, const char *msg) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << msg << ": cuFFT error code " << err << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate host memory for input and output
    cufftComplex *h_signal = (cufftComplex *)malloc(N * sizeof(cufftComplex));

    // Initialize input signal (real-valued for simplicity)
    for (int i = 0; i < N; i++) {
        h_signal[i].x = sinf(2.0f * M_PI * i / N);  // Real part
        h_signal[i].y = 0.0f;                      // Imaginary part
    }

    // Allocate device memory
    cufftComplex *d_signal;
    checkCudaError(cudaMalloc((void **)&d_signal, N * sizeof(cufftComplex)), "Failed to allocate device memory");

    // Copy input signal to device
    checkCudaError(cudaMemcpy(d_signal, h_signal, N * sizeof(cufftComplex), cudaMemcpyHostToDevice),
                   "Failed to copy signal to device");

    // Create FFT plan
    cufftHandle plan;
    checkCufftError(cufftPlan1d(&plan, N, CUFFT_C2C, 1), "Failed to create FFT plan");

    // Execute forward FFT
    std::cout << "Performing forward FFT...\n";
    checkCufftError(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD), "Failed to execute forward FFT");

    // Copy FFT result back to host
    checkCudaError(cudaMemcpy(h_signal, d_signal, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
                   "Failed to copy FFT result back to host");

    // Print some FFT results
    std::cout << "FFT result (first 10 points):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_signal[i].x << " + " << h_signal[i].y << "i\n";
    }

    // Execute inverse FFT
    std::cout << "\nPerforming inverse FFT...\n";
    checkCufftError(cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE), "Failed to execute inverse FFT");

    // Copy inverse FFT result back to host
    checkCudaError(cudaMemcpy(h_signal, d_signal, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
                   "Failed to copy inverse FFT result back to host");

    // Scale the inverse FFT results
    for (int i = 0; i < N; i++) {
        h_signal[i].x /= N;  // Scale real part
        h_signal[i].y /= N;  // Scale imaginary part
    }

    // Print some inverse FFT results
    std::cout << "Inverse FFT result (first 10 points):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_signal[i].x << " + " << h_signal[i].y << "i\n";
    }

    // Clean up
    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);

    return 0;
}
```

---

### **Key Points**
1. **Input Signal**: Real-valued sinusoidal signal is initialized for simplicity.
2. **FFT Execution**: `cufftExecC2C` performs both forward and inverse transforms.
3. **Scaling**: After the inverse FFT, results must be divided by the transform size.

---

### **How to Run the Code**
1. Compile the code with the cuFFT library:  
   ```bash
   nvcc -o fft_example fft_example.cu -lcufft
   ```
2. Run the executable:  
   ```bash
   ./fft_example
   ```

---

### **Output**  
You will see the transformed FFT results (complex-valued) and the scaled inverse FFT results, which match the original signal.

This example showcases the power and ease of using cuFFT for high-performance Fourier transforms on the GPU. Let me know if you need further assistance or additional examples!