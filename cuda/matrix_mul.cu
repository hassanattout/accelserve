#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

constexpr int TILE = 16;

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

void checkCublas(
    cublasStatus_t result,
    const char* call,
    const char* file,
    int line
) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr
            << "cuBLAS error\n"
            << "Call: " << call
            << "\nFile: " << file
            << ":" << line
            << "\n";

        std::exit(EXIT_FAILURE);
    }
}

#define CUBLAS_CHECK(call) \
    checkCublas((call), #call, __FILE__, __LINE__)

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 0) {
        return (
            values[n / 2 - 1]
            + values[n / 2]
        ) / 2.0;
    }

    return values[n / 2];
}

__attribute__((noinline))
void cpuMatrixMul(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {

            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                sum +=
                    A[row * N + k]
                    *
                    B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

__global__ void matrixMulNaive(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    const int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (row < N && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum +=
                A[row * N + k]
                *
                B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void matrixMulTiled(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col =
        blockIdx.x * TILE + tx;

    const int row =
        blockIdx.y * TILE + ty;

    float sum = 0.0f;

    const int numberOfTiles =
        (N + TILE - 1) / TILE;

    for (int tile = 0;
         tile < numberOfTiles;
         tile++) {

        const int aCol =
            tile * TILE + tx;

        const int bRow =
            tile * TILE + ty;

        if (row < N && aCol < N) {
            tileA[ty][tx] =
                A[row * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (bRow < N && col < N) {
            tileB[ty][tx] =
                B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum +=
                tileA[ty][k]
                *
                tileB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512;

    const int CPU_RUNS = 3;
    const int WARMUP_RUNS = 5;
    const int GPU_RUNS = 20;

    const size_t elements =
        static_cast<size_t>(N) * N;

    const size_t bytes =
        elements * sizeof(float);

    std::vector<float> A(elements, 1.0f);
    std::vector<float> B(elements, 1.0f);

    std::vector<float> C_cpu(elements);
    std::vector<float> C_naive(elements);
    std::vector<float> C_tiled(elements);
    std::vector<float> C_cublas(elements);

    // CPU
    std::vector<double> cpuTimes;

    for (int run = 0; run < CPU_RUNS; run++) {

        auto start =
            std::chrono::high_resolution_clock::now();

        cpuMatrixMul(
            A.data(),
            B.data(),
            C_cpu.data(),
            N
        );

        auto end =
            std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli>
            elapsed = end - start;

        cpuTimes.push_back(elapsed.count());
    }

    const double cpuMedian =
        median(cpuTimes);

    // GPU setup
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

    dim3 threads(TILE, TILE);

    dim3 blocks(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // -------------------------
    // Naive CUDA
    // -------------------------

    for (int i = 0; i < WARMUP_RUNS; i++) {
        matrixMulNaive<<<blocks, threads>>>(
            d_A, d_B, d_C, N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> naiveTimes;

    for (int i = 0; i < GPU_RUNS; i++) {

        CUDA_CHECK(cudaEventRecord(startEvent));

        matrixMulNaive<<<blocks, threads>>>(
            d_A, d_B, d_C, N
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                startEvent,
                stopEvent
            )
        );

        naiveTimes.push_back(ms);
    }

    CUDA_CHECK(
        cudaMemcpy(
            C_naive.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    // -------------------------
    // Tiled CUDA
    // -------------------------

    for (int i = 0; i < WARMUP_RUNS; i++) {
        matrixMulTiled<<<blocks, threads>>>(
            d_A, d_B, d_C, N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> tiledTimes;

    for (int i = 0; i < GPU_RUNS; i++) {

        CUDA_CHECK(cudaEventRecord(startEvent));

        matrixMulTiled<<<blocks, threads>>>(
            d_A, d_B, d_C, N
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                startEvent,
                stopEvent
            )
        );

        tiledTimes.push_back(ms);
    }

    CUDA_CHECK(
        cudaMemcpy(
            C_tiled.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    // -------------------------
    // cuBLAS
    // -------------------------

    cublasHandle_t handle;

    CUBLAS_CHECK(
        cublasCreate(&handle)
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < WARMUP_RUNS; i++) {

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,

                d_B,
                N,

                d_A,
                N,

                &beta,

                d_C,
                N
            )
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> cublasTimes;

    for (int i = 0; i < GPU_RUNS; i++) {

        CUDA_CHECK(cudaEventRecord(startEvent));

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,

                d_B,
                N,

                d_A,
                N,

                &beta,

                d_C,
                N
            )
        );

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                startEvent,
                stopEvent
            )
        );

        cublasTimes.push_back(ms);
    }

    CUDA_CHECK(
        cudaMemcpy(
            C_cublas.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    // Validation
    const float expected =
        static_cast<float>(N);

    auto validate =
        [&](const std::vector<float>& C) {

            for (float value : C) {
                if (value != expected) {
                    return false;
                }
            }

            return true;
        };

    const bool cpuValid =
        validate(C_cpu);

    const bool naiveValid =
        validate(C_naive);

    const bool tiledValid =
        validate(C_tiled);

    const bool cublasValid =
        validate(C_cublas);

    // Statistics
    const double naiveMedian =
        median(naiveTimes);

    const double tiledMedian =
        median(tiledTimes);

    const double cublasMedian =
        median(cublasTimes);

    const double operations =
        2.0
        * static_cast<double>(N)
        * N
        * N;

    auto gflops =
        [&](double milliseconds) {
            return operations /
                   (milliseconds * 1e6);
        };

    // Results
    std::cout
        << "AccelServe GEMM Benchmark\n";

    std::cout
        << "=========================\n";

    std::cout
        << "GPU: "
        << properties.name
        << "\n";

    std::cout
        << "Matrix: "
        << N << " x " << N
        << "\n\n";

    std::cout
        << "CPU validation:    "
        << (cpuValid ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "Naive validation:  "
        << (naiveValid ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "Tiled validation:  "
        << (tiledValid ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "cuBLAS validation: "
        << (cublasValid ? "PASSED" : "FAILED")
        << "\n\n";

    std::cout
        << "CPU median:    "
        << cpuMedian
        << " ms\n";

    std::cout
        << "Naive CUDA:    "
        << naiveMedian
        << " ms\n";

    std::cout
        << "Tiled CUDA:    "
        << tiledMedian
        << " ms\n";

    std::cout
        << "cuBLAS SGEMM:  "
        << cublasMedian
        << " ms\n\n";

    std::cout
        << "Naive:  "
        << gflops(naiveMedian)
        << " GFLOP/s\n";

    std::cout
        << "Tiled:  "
        << gflops(tiledMedian)
        << " GFLOP/s\n";

    std::cout
        << "cuBLAS: "
        << gflops(cublasMedian)
        << " GFLOP/s\n\n";

    std::cout
        << "cuBLAS vs naive: "
        << naiveMedian / cublasMedian
        << "x\n";

    std::cout
        << "cuBLAS vs tiled: "
        << tiledMedian / cublasMedian
        << "x\n";

    CUBLAS_CHECK(
        cublasDestroy(handle)
    );

    CUDA_CHECK(
        cudaEventDestroy(startEvent)
    );

    CUDA_CHECK(
        cudaEventDestroy(stopEvent)
    );

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}