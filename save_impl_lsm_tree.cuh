#ifndef IMPL_LSM_TREE_CUH
#define IMPL_LSM_TREE_CUH


#include "definitions_coarse_granular.cuh"
#include "definitions_opt.cuh"
#include "memory_layout.cuh"

#include <cmath>

#include <cub/cub.cuh>

//namespace cub {
#include <cub/device/device_merge.cuh>

template <typename key_type, smallsize chunk_size, smallsize max_cached_levels>
GLOBALQUALIFIER
void lsm_lookup_kernel(
        const key_type* level_keys,
        const smallsize* level_values,
        const key_type* staged_keys,
        const smallsize* staged_values,
        const key_type* query_keys,
        smallsize* results,
        smallsize size,
        smallsize level_count,
        smallsize inserted_chunk_counter,
        smallsize staged_size
) {
    constexpr smallsize max_cached_entries = smallsize(1) << max_cached_levels;
    __shared__ key_type top_level_keys[max_cached_entries];

    const smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
    const smallsize local_tid = threadIdx.x;

    key_type lookup;
    bool to_find = false;
    if (tid < size) {
        lookup = query_keys[tid];
        to_find = true;
    }
    smallsize result = not_found;

    if (to_find && staged_size > 0) {
        smallsize staged_offset = opt_reverse_device_binary_search<key_type, true>(
                (const uint8_t*) staged_keys, staged_size, sizeof(key_type), lookup);
        if (staged_keys[staged_offset] == lookup) {
            result = staged_values[staged_offset];
            to_find = false;
        }
    }

    for (int k = 0; k < level_count; ++k) {
        if ((inserted_chunk_counter & (1u << k)) == 0) continue;

        smallsize num_element_level_k = chunk_size << k;
        smallsize offset = num_element_level_k - chunk_size;

        const auto current_level_keys = level_keys + offset;

        smallsize search_levels = ilog2_gpu(num_element_level_k);
        smallsize cached_levels = min(max_cached_levels, search_levels);
        smallsize cached_stride = smallsize(1) << (search_levels - cached_levels);
        smallsize cached_entries_count = SDIV(num_element_level_k, cached_stride);
        smallsize cached_offset = num_element_level_k - 1 - cached_stride * (cached_entries_count - 1);

        __syncthreads();

        // load top keys
        if (local_tid < cached_entries_count) {
            top_level_keys[local_tid] = current_level_keys[cached_offset + local_tid * cached_stride];
        }

        __syncthreads();

        if (!to_find) continue;

        smallsize coarse_offset = opt_reverse_device_binary_search<key_type, true>(
            (const uint8_t*) top_level_keys, cached_entries_count, sizeof(key_type), lookup);

        smallsize continue_offset = num_element_level_k - 1 - (cached_stride * (cached_entries_count - 1 - coarse_offset));
        smallsize continue_stride = cached_stride >> 1u;

        smallsize final_offset = opt_reverse_device_binary_search<key_type, true>(
            (const uint8_t*) current_level_keys, sizeof(key_type), lookup, continue_offset, continue_stride);

        //smallsize final_offset = opt_reverse_device_binary_search<key_type, true>(
        //    (const uint8_t*) current_level_keys, num_element_level_k, sizeof(key_type), lookup);

        if (current_level_keys[final_offset] == lookup) {
            result = level_values[offset + final_offset];
            to_find = false;
        }
    }
    __syncthreads();

    if (tid < size) {
        results[tid] = result;
    }
}


template <typename key_type, typename value_type>
void untimed_pair_merge(
        void* temp, size_t temp_bytes,
        const key_type* k1, const key_type* k2, key_type* ko,
        const value_type* v1, const value_type* v2, value_type* vo,
        size_t input_size1, size_t input_size2, cudaStream_t stream) {
    cub::DeviceMerge::MergePairs(
        temp, temp_bytes,
        k1, v1, input_size1,
        k2, v2, input_size2,
        ko, vo,
        {},
        stream);
}


template <typename key_type, typename value_type>
size_t find_pair_merge_buffer_size(size_t input_size1, size_t input_size2) {
    size_t temp_bytes_required = 0;
    cub::DeviceMerge::MergePairs(
        nullptr, temp_bytes_required,
        (key_type*)nullptr, (value_type*)nullptr, input_size1,
        (key_type*)nullptr, (value_type*)nullptr, input_size2,
        (key_type*)nullptr, (value_type*)nullptr);
    return temp_bytes_required;
}


template <typename key_type_, size_t chunk_size = 8>
class lsm_tree_ashkiani final {
public:
    using key_type = key_type_;

    // taken from original code
    static constexpr size_t threads_per_block = 256;
    static constexpr size_t max_cached_levels = 8;

private:
    cuda_buffer<key_type> level_keys_buffer;
    cuda_buffer<smallsize> level_values_buffer;
    cuda_buffer<key_type> staging_keys_buffer;
    cuda_buffer<smallsize> staging_values_buffer;

    cuda_buffer<uint8_t> temp_sort_buffer;
    cuda_buffer<key_type> temp_keys_buffer_a;
    cuda_buffer<smallsize> temp_values_buffer_a;
    cuda_buffer<key_type> temp_keys_buffer_b;
    cuda_buffer<smallsize> temp_values_buffer_b;

    size_t total_available_slots = 0;
    size_t level_count = 0;

    size_t staged_insert_size = 0;
    size_t inserted_chunk_counter = 0;

    void merge_at_level(size_t initial_level, const key_type* insert_keys, const smallsize* insert_values, cudaStream_t stream) {
        size_t new_chunk_count = 1u << initial_level;
        size_t level_size = chunk_size << initial_level;
        size_t level_offset = (level_size - chunk_size);

        auto source_keys = temp_keys_buffer_a.ptr();
        auto source_values = temp_values_buffer_a.ptr();
        auto dest_keys = temp_keys_buffer_b.ptr();
        auto dest_values = temp_values_buffer_b.ptr();

        size_t target_level = initial_level;
        while ((inserted_chunk_counter & (1u << target_level)) != 0) ++target_level;

        for (size_t level = initial_level; level < target_level; ++level) {
            //std::cerr << "merging with level " << level << " of size " << level_size << "\n";
            auto local_source_keys = level == initial_level ? insert_keys : source_keys;
            auto local_source_values = level == initial_level ? insert_values : source_values;
            untimed_pair_merge<key_type, smallsize>(
                    temp_sort_buffer.raw_ptr, temp_sort_buffer.size_in_bytes(),
                    local_source_keys, level_keys_buffer.ptr() + level_offset, dest_keys,
                    local_source_values, level_values_buffer.ptr() + level_offset, dest_values,
                    level_size, level_size, stream);
            level_offset += level_size;
            level_size *= 2;

            // swap temp buffers
            std::swap(source_keys, dest_keys);
            std::swap(source_values, dest_values);
        }

        //std::cerr << "writing result to level " << target_level << "\n";
        cudaMemcpyAsync(
                level_keys_buffer.ptr() + level_offset,
                initial_level == target_level ? insert_keys : source_keys,
                level_size * sizeof(key_type),
                cudaMemcpyDefault, stream);
        cudaMemcpyAsync(
                level_values_buffer.ptr() + level_offset,
                initial_level == target_level ? insert_values : source_values,
                level_size * sizeof(smallsize),
                cudaMemcpyDefault, stream);

        inserted_chunk_counter += new_chunk_count;
    }

public:

    static constexpr const char* name = "lsm_tree";
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_lower_bound_rank = operation_support::none;
    static constexpr operation_support can_multi_lookup = operation_support::none;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_update = operation_support::async;

    static std::string short_description() {
        return std::string("lsm_tree_ashkiani");
    }

    static parameters_type parameters() {
        return {
                {"chunk_size", std::to_string(chunk_size)},
                {"threads_per_block", std::to_string(threads_per_block)},
                {"max_cached_levels", std::to_string(max_cached_levels)},
        };
    }

    size_t gpu_resident_bytes() {
        return level_keys_buffer.size_in_bytes() + level_values_buffer.size_in_bytes() +
               staging_keys_buffer.size_in_bytes() + staging_values_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {

        level_count = 1;
        total_available_slots = chunk_size;
        while (true) {
            if (total_available_slots >= max_size) break;
            level_count++;
            total_available_slots = (chunk_size << level_count) - chunk_size;
        }

        level_keys_buffer.alloc(total_available_slots); C2EX
        level_values_buffer.alloc(total_available_slots); C2EX
        staging_keys_buffer.alloc(chunk_size); C2EX
        staging_values_buffer.alloc(chunk_size); C2EX

        {
            size_t max_level_size = (chunk_size << (level_count - 1));
            size_t max_temp_sort_bytes = std::max(
                    find_pair_sort_buffer_size<key_type, smallsize>(max_level_size),
                    find_pair_merge_buffer_size<key_type, smallsize>(max_level_size, max_level_size));
            temp_sort_buffer.alloc(max_temp_sort_bytes); C2EX

            temp_keys_buffer_a.alloc(max_level_size); C2EX
            temp_values_buffer_a.alloc(max_level_size); C2EX
            temp_keys_buffer_b.alloc(max_level_size); C2EX
            temp_values_buffer_b.alloc(max_level_size); C2EX
        }

        // we want to report only the space we actually need during build
        size_t temp_storage_bytes = 0;
        {
            init_offsets(temp_values_buffer_a.ptr(), size, build_time_ms);

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            timed_pair_sort(
                temp_sort_buffer.raw_ptr, temp_storage_bytes,
                keys, level_keys_buffer.ptr(), temp_values_buffer_a.ptr(), level_values_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize(); C2EX
        };

        inserted_chunk_counter = size / chunk_size;
        staged_insert_size = size % chunk_size;

        size_t end_of_data_block = size;

        // copy last (incomplete) chunk to staging buffer
        if (staged_insert_size > 0) {
            //std::cerr << "copying last " << staged_insert_size << " elements to temp buffer\n";
            cudaMemcpyAsync(
                    staging_keys_buffer.ptr(),
                    level_keys_buffer.ptr() + end_of_data_block - staged_insert_size,
                    staged_insert_size * sizeof(key_type),
                    cudaMemcpyDefault);
            cudaMemcpyAsync(
                    staging_values_buffer.ptr(),
                    level_values_buffer.ptr() + end_of_data_block - staged_insert_size,
                    staged_insert_size * sizeof(smallsize),
                    cudaMemcpyDefault);
            end_of_data_block -= staged_insert_size;
        }

        // fill the main levels correctly
        for (size_t rev_shift = 0; rev_shift < 32; ++rev_shift) {
            auto shift = 31 - rev_shift;
            if ((inserted_chunk_counter & (1u << shift)) == 0) continue;
            size_t level_size = chunk_size << shift;
            size_t level_offset = (level_size - chunk_size);
            size_t source_offset = end_of_data_block - level_size;
            end_of_data_block -= level_size;
            //std::cerr << "copying level " << shift << " of size " << level_size << " from " << source_offset << " to " << level_offset << "\n";
            if (level_offset == source_offset) continue;
            cudaMemcpyAsync(
                    temp_keys_buffer_b.ptr(),
                    level_keys_buffer.ptr() + source_offset,
                    level_size * sizeof(key_type),
                    cudaMemcpyDefault);
            cudaMemcpyAsync(
                    level_keys_buffer.ptr() + level_offset,
                    temp_keys_buffer_b.ptr(),
                    level_size * sizeof(key_type),
                    cudaMemcpyDefault);
            cudaMemcpyAsync(
                    temp_values_buffer_b.ptr(),
                    level_values_buffer.ptr() + source_offset,
                    level_size * sizeof(smallsize),
                    cudaMemcpyDefault);
            cudaMemcpyAsync(
                    level_values_buffer.ptr() + level_offset,
                    temp_values_buffer_b.ptr(),
                    level_size * sizeof(smallsize),
                    cudaMemcpyDefault);
        }

        if (build_bytes) *build_bytes += gpu_resident_bytes() + sizeof(smallsize) * size + temp_storage_bytes;
    }

    void dump_tree() {
        size_t n = chunk_size;
        size_t o = 0;
        size_t k = 0;
        for (; k < level_count; ++k) {
            if ((inserted_chunk_counter & (1u << k)) == 0) continue;
            n = chunk_size << k;
            o = n - chunk_size;
            std::cerr << "LEVEL " << k << ": ";
            level_keys_buffer.debug_dump(n, o);
        }
        std::cerr << "OVERFLOW : ";
        staging_keys_buffer.debug_dump(staged_insert_size);
    }

    void destroy() {
        level_keys_buffer.free();
        level_values_buffer.free();
        temp_sort_buffer.free();
        temp_keys_buffer_a.free();
        temp_values_buffer_a.free();
        temp_keys_buffer_b.free();
        temp_values_buffer_b.free();
        total_available_slots = 0;
        staged_insert_size = 0;
        inserted_chunk_counter = 0;
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        lsm_lookup_kernel
                <key_type, chunk_size, max_cached_levels>
                <<<SDIV(size, threads_per_block), threads_per_block, 0, stream>>>(
            level_keys_buffer.ptr(),
            level_values_buffer.ptr(),
            staging_keys_buffer.ptr(),
            staging_values_buffer.ptr(),
            keys,
            result,
            size,
            level_count,
            inserted_chunk_counter,
            staged_insert_size);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {}
    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        // todo
    }

    void insert(const key_type* insert_list, const smallsize* positions, size_t size, cudaStream_t stream) {

        size_t insert_list_offset = 0;

        size_t newly_inserted_chunk_counter = size / chunk_size;
        size_t new_staged_insert_size = size % chunk_size;

        if (new_staged_insert_size > 0) {
            // move overflowing elements to staging buffer
            size_t remaining_slots = chunk_size - staged_insert_size;
            // copy to staging buffer next to the already existing data
            cudaMemcpyAsync(
                    staging_keys_buffer.ptr() + staged_insert_size,
                    insert_list,
                    remaining_slots * sizeof(key_type),
                    cudaMemcpyDefault, stream);
            cudaMemcpyAsync(
                    staging_values_buffer.ptr() + staged_insert_size,
                    positions,
                    remaining_slots * sizeof(smallsize),
                    cudaMemcpyDefault, stream);
            // sort to temp buffer a
            untimed_pair_sort(
                    temp_sort_buffer.raw_ptr, temp_sort_buffer.size_in_bytes(),
                    staging_keys_buffer.ptr(), temp_keys_buffer_a.ptr(),
                    staging_values_buffer.ptr(), temp_values_buffer_a.ptr(),
                    chunk_size, stream);

            if (new_staged_insert_size + staged_insert_size >= chunk_size) {
                // merge with existing levels
                merge_at_level(0, temp_keys_buffer_a.ptr(), temp_values_buffer_a.ptr(), stream);
                insert_list_offset += remaining_slots;

                size_t remaining_size = new_staged_insert_size + staged_insert_size - chunk_size;
                if (remaining_size > 0) {
                    // copy remaining elements to staging buffer
                    cudaMemcpyAsync(
                            staging_keys_buffer.ptr(),
                            insert_list + insert_list_offset,
                            remaining_size * sizeof(key_type),
                            cudaMemcpyDefault, stream);
                    cudaMemcpyAsync(
                            staging_values_buffer.ptr(),
                            positions + insert_list_offset,
                            remaining_size * sizeof(smallsize),
                            cudaMemcpyDefault, stream);
                    insert_list_offset += remaining_size;
                }
                staged_insert_size = remaining_size;
            } else {
                // copy back to staging buffer
                cudaMemcpyAsync(
                        staging_keys_buffer.ptr(),
                        temp_keys_buffer_a.ptr(),
                        (new_staged_insert_size + staged_insert_size) * sizeof(key_type),
                        cudaMemcpyDefault, stream);
                cudaMemcpyAsync(
                        staging_values_buffer.ptr(),
                        temp_values_buffer_a.ptr(),
                        (new_staged_insert_size + staged_insert_size) * sizeof(smallsize),
                        cudaMemcpyDefault, stream);
                staged_insert_size += new_staged_insert_size;
                insert_list_offset += new_staged_insert_size;
            }
        }

        for (size_t new_level = 0; new_level < level_count; ++new_level) {
            if ((newly_inserted_chunk_counter & (1u << new_level)) == 0) continue;
            merge_at_level(new_level, insert_list + insert_list_offset, positions + insert_list_offset, stream);
            insert_list_offset += chunk_size << new_level;
        }
    }

    void remove(const key_type* delete_list, size_t size, cudaStream_t stream) {
        // authors did not provide an implementation
    }
};


//}
#endif
