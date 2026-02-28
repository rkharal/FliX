#ifndef UTILITIES_H
#define UTILITIES_H

#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_set>
#include <optional>

#include <cub/cub.cuh>

#include <thrust/shuffle.h>
#include <thrust/sort.h>

#include "cuda_buffer.cuh"


template <typename key_type>
size_t footprint(size_t element_count) {
    return element_count * sizeof(key_type);
}


class cuda_timer final {
    cudaEvent_t timerstart, timerstop;
    cudaStream_t stream;

public:
    cuda_timer(cudaStream_t stream) : stream(stream) {
        cudaEventCreate(&timerstart);
        cudaEventCreate(&timerstop);
    }
    ~cuda_timer() {
        cudaEventDestroy(timerstart);
        cudaEventDestroy(timerstop);
    }
    void start() {
        cudaEventRecord(timerstart, stream);
    }
    void stop() {
        cudaEventRecord(timerstop, stream);
    }
    float time_ms() {
        float timerdelta;
        cudaEventSynchronize(timerstop);
        cudaEventElapsedTime(&timerdelta, timerstart, timerstop);
        return timerdelta;
    }
};


class scoped_cuda_timer final {
    cuda_timer timer;
    double* result_location;

public:
    scoped_cuda_timer(cudaStream_t stream, double* result_location) : timer(stream), result_location(result_location) {
        if (result_location) {
            timer.start();
        }
    }
    ~scoped_cuda_timer() {
        if (result_location) {
            timer.stop();
            *result_location += timer.time_ms();
        }
    }
};


void rti_assert(bool predicate, const std::string& desc = {}) {
    if (!predicate) throw std::runtime_error(desc);
}


template <typename value_type>
void show_vector(const std::vector<value_type>& local_buffer, size_t max_output = std::numeric_limits<size_t>::max()) {
    for (size_t i = 0; i < std::min(max_output, local_buffer.size()); ++i) {
        std::cout << local_buffer[i] << "  ";
    }
    std::cout << std::endl;
}


template <typename key_type>
void check_result(const std::vector<key_type>& lower, const std::vector<key_type>& upper, const std::vector<smallsize>& expected, cuda_buffer<smallsize>& result_buffer) {
    std::vector<smallsize> test_output(expected.size());
    result_buffer.download(test_output.data(), test_output.size());
    for (size_t i = 0; i < expected.size(); ++i) {
       if (expected[i] != test_output[i]) {
           std::cerr << "data mismatch at index " << i << " for range " << lower[i] << "-" << upper[i] << ": expected " << expected[i] << ", but received " << test_output[i] << std::endl;
           throw std::logic_error("stop");
       }
    }
}


template <typename key_type>
size_t find_sort_buffer_size(size_t input_size) {
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes_required, (key_type*)nullptr, (key_type*)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}


template <typename key_type, typename value_type>
size_t find_pair_sort_buffer_size(size_t input_size) {
    size_t temp_bytes_required = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes_required, (key_type*)nullptr, (key_type*)nullptr, (value_type*)nullptr, (value_type*)nullptr, input_size, 0, sizeof(key_type) * 8);
    return temp_bytes_required;
}


template <typename key_type>
void untimed_sort(void* temp, size_t temp_bytes, const key_type* input, key_type* output, size_t input_size) {
    // future work: use a different algo for small input buffers
    // maybe check out cub::BlockRadixSort or cub::StableOddEvenSort
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, input, output, input_size, 0, sizeof(key_type) * 8);
}


template <typename key_type>
void timed_sort(void* temp, size_t temp_bytes, const key_type* input, key_type* output, size_t input_size, double* time_ms) {
    scoped_cuda_timer timer(0, time_ms);
    untimed_sort(temp, temp_bytes, input, output, input_size);
}


template <typename key_type, typename value_type>
void untimed_pair_sort(void* temp, size_t temp_bytes, const key_type* ki, key_type* ko, const value_type* vi, value_type* vo, size_t input_size) {
    cub::DeviceRadixSort::SortPairs(temp, temp_bytes, ki, ko, vi, vo, input_size, 0, sizeof(key_type) * 8);
}


template <typename key_type, typename value_type>
void timed_pair_sort(void* temp, size_t temp_bytes, const key_type* ki, key_type* ko, const value_type* vi, value_type* vo, size_t input_size, double* time_ms) {
    scoped_cuda_timer timer(0, time_ms);
    untimed_pair_sort(temp, temp_bytes, ki, ko, vi, vo, input_size);
}


GLOBALQUALIFIER
void init_offsets_kernel(smallsize* buffer, smallsize size, smallsize first_offset) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size) return;
    buffer[tid] = static_cast<smallsize>(tid) + first_offset;
}


void init_offsets(smallsize* buffer, size_t size, size_t first_offset, double* time_ms) {
    scoped_cuda_timer timer(0, time_ms);
    init_offsets_kernel<<<SDIV(size, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(buffer, size, first_offset);
}


void init_offsets(smallsize* buffer, size_t size, double* time_ms) {
    init_offsets(buffer, size, 0, time_ms);
}

#endif
