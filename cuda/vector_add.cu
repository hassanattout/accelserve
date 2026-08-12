#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

void checkCuda(
    cudaError_t result,
    const char* call,
    const char* file,
    int line
) {
    if (result != cudaSuccess) {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(result)
            << "\nCall: " << call
            << "\nFile: " << file
            << ":" << line
            << "\n";

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) \
    checkCuda((call), #call, __FILE__, __LINE__)

__attribute__((noinline))
void cpuAccumulate(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    for (int i = 0; i < N; i++) {
        C[i] += A[i] + B[i];
    }
}

__global__ void gpuAccumulate(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N) {
        C[i] += A[i] + B[i];
    }
}

int main() {
    const int N = 1'000'000;
    const int ITERATIONS = 100;

    const size_t bytes =
        N * sizeof(float);

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);

    std::vector<float> C_cpu(N, 0.0f);
    std::vector<float> C_gpu(N, 0.0f);

    // ================================
    // CPU
    // ================================

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    for (int iteration = 0;
         iteration < ITERATIONS;
         iteration++) {

        cpuAccumulate(
            A.data(),
            B.data(),
            C_cpu.data(),
            N
        );
    }

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        cpuElapsed = cpuEnd - cpuStart;

    // ================================
    // GPU setup
    // ================================

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            0
        )
    );

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    const int threadsPerBlock = 256;

    const int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // ================================
    // GPU end-to-end start
    // ================================

    auto gpuTotalStart =
        std::chrono::high_resolution_clock::now();

    // Copy inputs only ONCE
    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_C,
            0,
            bytes
        )
    );

    // ================================
    // Measure 100 GPU kernel launches
    // ================================

    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    CUDA_CHECK(
        cudaEventCreate(&kernelStart)
    );

    CUDA_CHECK(
        cudaEventCreate(&kernelStop)
    );

    CUDA_CHECK(
        cudaEventRecord(kernelStart)
    );

    for (int iteration = 0;
         iteration < ITERATIONS;
         iteration++) {

        gpuAccumulate<<<
            blocksPerGrid,
            threadsPerBlock
        >>>(
            d_A,
            d_B,
            d_C,
            N
        );

        CUDA_CHECK(
            cudaGetLastError()
        );
    }

    CUDA_CHECK(
        cudaEventRecord(kernelStop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(kernelStop)
    );

    float gpuKernelMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &gpuKernelMs,
            kernelStart,
            kernelStop
        )
    );

    // Copy result back only ONCE
    CUDA_CHECK(
        cudaMemcpy(
            C_gpu.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    auto gpuTotalEnd =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        gpuTotalElapsed =
            gpuTotalEnd - gpuTotalStart;

    // ================================
    // Validation
    // ================================

    const float expected =
        ITERATIONS * 3.0f;

    bool cpuValid = true;
    bool gpuValid = true;

    for (int i = 0; i < N; i++) {
        if (C_cpu[i] != expected) {
            cpuValid = false;
            break;
        }
    }

    for (int i = 0; i < N; i++) {
        if (C_gpu[i] != expected) {
            gpuValid = false;
            break;
        }
    }

    // ================================
    // Results
    // ================================

    const double cpuMs =
        cpuElapsed.count();

    const double gpuEndToEndMs =
        gpuTotalElapsed.count();

    const double kernelSpeedup =
        cpuMs / gpuKernelMs;

    const double endToEndSpeedup =
        cpuMs / gpuEndToEndMs;

    std::cout
        << "AccelServe GPU-Resident Workload Benchmark\n";

    std::cout
        << "=========================================\n";

    std::cout
        << "GPU: "
        << properties.name
        << "\n";

    std::cout
        << "Elements: "
        << N
        << "\n";

    std::cout
        << "Iterations: "
        << ITERATIONS
        << "\n";

    std::cout
        << "Expected result: "
        << expected
        << "\n\n";

    std::cout
        << "CPU validation: "
        << (cpuValid ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "GPU validation: "
        << (gpuValid ? "PASSED" : "FAILED")
        << "\n\n";

    std::cout
        << "CPU total:       "
        << cpuMs
        << " ms\n";

    std::cout
        << "GPU kernels:     "
        << gpuKernelMs
        << " ms\n";

    std::cout
        << "GPU end-to-end:  "
        << gpuEndToEndMs
        << " ms\n\n";

    std::cout
        << "Kernel speedup:  "
        << kernelSpeedup
        << "x\n";

    std::cout
        << "End-end speedup: "
        << endToEndSpeedup
        << "x\n";

    CUDA_CHECK(
        cudaEventDestroy(kernelStart)
    );

    CUDA_CHECK(
        cudaEventDestroy(kernelStop)
    );

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}