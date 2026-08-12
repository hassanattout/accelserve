#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    const int N = 1'000'000;
    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 20;

    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    // Get pointers to the vectors' underlying memory
    const float* ptrA = A.data();
    const float* ptrB = B.data();
    float* ptrC = C.data();

    // Warm-up runs
    for (int run = 0; run < WARMUP_RUNS; run++) {
        for (int i = 0; i < N; i++) {
            ptrC[i] = ptrA[i] + ptrB[i];
        }
    }

    std::vector<double> times;

    // Measured runs
    for (int run = 0; run < BENCHMARK_RUNS; run++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < N; i++) {
            ptrC[i] = ptrA[i] + ptrB[i];
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        times.push_back(elapsed.count());
    }

    // Validate result
    bool valid = true;

    for (int i = 0; i < N; i++) {
        if (ptrC[i] != 3.0f) {
            valid = false;
            break;
        }
    }

    // Calculate statistics
    double total =
        std::accumulate(times.begin(), times.end(), 0.0);

    double average = total / times.size();

    std::sort(times.begin(), times.end());

    double minimum = times.front();
    double maximum = times.back();
    double median = times[times.size() / 2];

    std::cout << "AccelServe CPU Vector Addition Benchmark\n";
    std::cout << "----------------------------------------\n";

    std::cout << "Elements: " << N << "\n";
    std::cout << "Runs: " << BENCHMARK_RUNS << "\n";

    std::cout << "Validation: "
              << (valid ? "PASSED" : "FAILED")
              << "\n";

    std::cout << "Minimum: " << minimum << " ms\n";
    std::cout << "Median:  " << median << " ms\n";
    std::cout << "Average: " << average << " ms\n";
    std::cout << "Maximum: " << maximum << " ms\n";

    return 0;
}