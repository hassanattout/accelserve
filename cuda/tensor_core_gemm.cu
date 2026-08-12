#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
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

int main() {
    const int N = 2048;
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 20;

    const size_t elements =
        static_cast<size_t>(N) * N;

    const size_t fp32Bytes =
        elements * sizeof(float);

    const size_t fp16Bytes =
        elements * sizeof(__half);

    std::vector<float> A_fp32(elements, 1.0f);
    std::vector<float> B_fp32(elements, 1.0f);

    std::vector<__half> A_fp16(elements);
    std::vector<__half> B_fp16(elements);

    for (size_t i = 0; i < elements; i++) {
        A_fp16[i] = __float2half(1.0f);
        B_fp16[i] = __float2half(1.0f);
    }

    float* d_A_fp32 = nullptr;
    float* d_B_fp32 = nullptr;
    float* d_C_fp32 = nullptr;

    __half* d_A_fp16 = nullptr;
    __half* d_B_fp16 = nullptr;

    float* d_C_tensor = nullptr;

    CUDA_CHECK(
        cudaMalloc(&d_A_fp32, fp32Bytes)
    );

    CUDA_CHECK(
        cudaMalloc(&d_B_fp32, fp32Bytes)
    );

    CUDA_CHECK(
        cudaMalloc(&d_C_fp32, fp32Bytes)
    );

    CUDA_CHECK(
        cudaMalloc(&d_A_fp16, fp16Bytes)
    );

    CUDA_CHECK(
        cudaMalloc(&d_B_fp16, fp16Bytes)
    );

    CUDA_CHECK(
        cudaMalloc(&d_C_tensor, fp32Bytes)
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A_fp32,
            A_fp32.data(),
            fp32Bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B_fp32,
            B_fp32.data(),
            fp32Bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A_fp16,
            A_fp16.data(),
            fp16Bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B_fp16,
            B_fp16.data(),
            fp16Bytes,
            cudaMemcpyHostToDevice
        )
    );

    cublasHandle_t handle;

    CUBLAS_CHECK(
        cublasCreate(&handle)
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(
        cudaEventCreate(&startEvent)
    );

    CUDA_CHECK(
        cudaEventCreate(&stopEvent)
    );

    // =========================================
    // FP32 cuBLAS benchmark
    // =========================================

    for (int run = 0;
         run < WARMUP_RUNS;
         run++) {

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,
                d_B_fp32,
                N,
                d_A_fp32,
                N,
                &beta,
                d_C_fp32,
                N
            )
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<double> fp32Times;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         run++) {

        CUDA_CHECK(
            cudaEventRecord(startEvent)
        );

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,
                d_B_fp32,
                N,
                d_A_fp32,
                N,
                &beta,
                d_C_fp32,
                N
            )
        );

        CUDA_CHECK(
            cudaEventRecord(stopEvent)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stopEvent)
        );

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                startEvent,
                stopEvent
            )
        );

        fp32Times.push_back(ms);
    }

    // =========================================
    // FP16 Tensor Core benchmark
    // =========================================

    for (int run = 0;
         run < WARMUP_RUNS;
         run++) {

        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,

                d_B_fp16,
                CUDA_R_16F,
                N,

                d_A_fp16,
                CUDA_R_16F,
                N,

                &beta,

                d_C_tensor,
                CUDA_R_32F,
                N,

                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    std::vector<double> tensorTimes;

    for (int run = 0;
         run < BENCHMARK_RUNS;
         run++) {

        CUDA_CHECK(
            cudaEventRecord(startEvent)
        );

        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                N,
                N,
                &alpha,

                d_B_fp16,
                CUDA_R_16F,
                N,

                d_A_fp16,
                CUDA_R_16F,
                N,

                &beta,

                d_C_tensor,
                CUDA_R_32F,
                N,

                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT
            )
        );

        CUDA_CHECK(
            cudaEventRecord(stopEvent)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stopEvent)
        );

        float ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                startEvent,
                stopEvent
            )
        );

        tensorTimes.push_back(ms);
    }

    const double fp32Median =
        median(fp32Times);

    const double tensorMedian =
        median(tensorTimes);

    const double operations =
        2.0
        * static_cast<double>(N)
        * N
        * N;

    const double fp32GFLOPS =
        operations /
        (fp32Median * 1e6);

    const double tensorGFLOPS =
        operations /
        (tensorMedian * 1e6);

    std::cout
        << "AccelServe Tensor Core GEMM Benchmark\n";

    std::cout
        << "=====================================\n";

    std::cout
        << "Matrix: "
        << N << " x " << N
        << "\n\n";

    std::cout
        << "FP32 cuBLAS median: "
        << fp32Median
        << " ms\n";

    std::cout
        << "FP16 Tensor GEMM:   "
        << tensorMedian
        << " ms\n\n";

    std::cout
        << "FP32 throughput: "
        << fp32GFLOPS
        << " GFLOP/s\n";

    std::cout
        << "FP16 throughput: "
        << tensorGFLOPS
        << " GFLOP/s\n\n";

    std::cout
        << "Tensor speedup: "
        << fp32Median / tensorMedian
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

    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));

    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_tensor));

    return 0;
}