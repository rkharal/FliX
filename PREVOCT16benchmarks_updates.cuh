#ifndef BENCHMARKS_UPDATES_CUH
#define BENCHMARKS_UPDATES_CUH

#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <vector>
#include <random>
#include <unordered_set>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>  // To check file existence
//#include "utilities.cuh"
#include "../../input-generation/input_generation.h"    // Include the correct Zipf class     
//#include "../../input-generation/input_generation.h"  // Include the correct Zipf class

//#define DEBUG_BENCHMARK_OUTPUT//
//#define REBUILD_ON
//#define COMBINE_INSERT_DELETE

#ifdef REBUILD_INTERVAL
constexpr smallsize rebuild_frequency = REBUILD_INTERVAL; //every round;
#else
constexpr smallsize rebuild_frequency = 1;
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <string>

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <random>
#include "input_generation.h"
#include "debug_definitions_updates.cuh"
#include "utilities.cuh"

#ifdef SHIFTED
constexpr bool shift_insert_range = true;
#else
constexpr bool shift_insert_range = false;
#endif



// --- kernels ---
GLOBALQUALIFIER void fill_seq_kernel(uint32_t* out, size_t n) {
    const size_t i = threadIdx.x + blockIdx.x * size_t(blockDim.x);
    if (i < n) out[i] = static_cast<uint32_t>(i);
}

template <typename T>
GLOBALQUALIFIER void scatter_perm_kernel(const T* __restrict__ in_sorted,
                                         const uint32_t* __restrict__ perm_sorted,
                                         T* __restrict__ out_original,
                                         size_t n) {
    const size_t i = threadIdx.x + blockIdx.x * size_t(blockDim.x);
    if (i < n) out_original[perm_sorted[i]] = in_sorted[i];
}

// --- API: key-only sort with out-params (no copies of cuda_buffer) ---
template <typename key_t>
void key_only_sort_device(const cuda_buffer<key_t>& d_keys_in,
                          size_t n,
                          cuda_buffer<key_t>& d_sorted_keys,      // OUT: size n
                          cuda_buffer<uint32_t>& d_perm_sorted,    // OUT: size n
                          cuda_buffer<uint8_t>& d_aux,             // OUT: scratch
                          cudaStream_t stream) {
    d_sorted_keys.alloc(n);
    d_perm_sorted.alloc(n);

    cuda_buffer<uint32_t> d_perm_in;
    d_perm_in.alloc(n);

    const int TPB = 256;
    const int GRD = static_cast<int>((n + TPB - 1) / TPB);
    fill_seq_kernel<<<GRD, TPB, 0, stream>>>(d_perm_in.ptr(), n);

    const size_t aux_size = find_pair_sort_buffer_size<key_t, uint32_t>(n);
    d_aux.alloc(aux_size);

    // Sort keys while carrying indices (keys-only externally)
    untimed_pair_sort(d_aux.ptr(), aux_size,
                      d_keys_in.ptr(),  d_sorted_keys.ptr(),
                      d_perm_in.ptr(),  d_perm_sorted.ptr(),
                      n, stream);
}

// --- API: scatter sorted values back to original order ---
template <typename T>
void scatter_sorted_to_original(const cuda_buffer<T>& d_vals_sorted,
                                const cuda_buffer<uint32_t>& d_perm_sorted,
                                cuda_buffer<T>& d_out_original, // size n, pre-alloc or alloc(n)
                                size_t n,
                                cudaStream_t stream) {
    d_out_original.alloc(n); // safe if already allocated; otherwise alloc now
    const int TPB = 256;
    const int GRD = static_cast<int>((n + TPB - 1) / TPB);
    scatter_perm_kernel<<<GRD, TPB, 0, stream>>>(
        d_vals_sorted.ptr(), d_perm_sorted.ptr(), d_out_original.ptr(), n);
}

template <typename element_type>
void micro_benchmark_search_and_coalesced() {
    rc::result_collector rc_sort;
    rc::result_collector rc_coalesced;

    constexpr size_t runs = 20;

    size_t build_size_log = 28u;
    size_t lookup_size_log = 16u;
    size_t build_size = size_t(1) << build_size_log;
    size_t lookup_size = size_t(1) << lookup_size_log;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<element_type> elements(build_size);
    for (size_t i = 0; i < build_size; i++) {
        elements.push_back(static_cast<element_type>(i));
    }

    cuda_buffer<element_type> elements_buffer, dummy_buffer, output_elements_buffer, output_dummy_buffer;
    elements_buffer.alloc_and_upload(elements);
    dummy_buffer.alloc_and_upload(elements);
    output_elements_buffer.alloc_for_size(elements);
    output_dummy_buffer.alloc_for_size(elements);

    cuda_buffer<uint8_t> aux_buffer;
    size_t aux_size = find_pair_sort_buffer_size<element_type, element_type>(build_size);
    aux_buffer.alloc(aux_size);

    auto elements_ptr = elements_buffer.ptr();
    auto dummy_ptr = dummy_buffer.ptr();
    auto output_elements_ptr = output_elements_buffer.ptr();
    auto output_dummy_ptr = output_dummy_buffer.ptr();
    auto aux_ptr = aux_buffer.ptr();

    for (size_t sort_size_log = 8; sort_size_log <= build_size_log; sort_size_log++) {
        size_t sort_size = size_t(1) << sort_size_log;
        double time_ms = 0;
        {
            scoped_cuda_timer timer(stream, &time_ms);
            for (size_t i = 0; i < runs; i++) {
                untimed_pair_sort(aux_ptr, aux_size, elements_ptr, output_elements_ptr, dummy_ptr, output_dummy_ptr, sort_size, stream);
            }
        }
        rc::auto_commit_result(rc_sort)
            .add_parameter("key_size", sizeof(element_type))
            .add_parameter("sort_size_log", sort_size_log)
            .add_measurement("time_ms", time_ms / runs);
    }

    for (size_t threads_per_block_log = 5; threads_per_block_log <= 10; threads_per_block_log++) {
        size_t threads_per_block = size_t(1) << threads_per_block_log;

        for (size_t group_size_log = 0; group_size_log <= 10; group_size_log++) {
            size_t group_size = size_t(1) << group_size_log;
            double time_ms = 0;
            {
                scoped_cuda_timer timer(stream, &time_ms);
                for (size_t i = 0; i < runs; i++) {
                    lambda_kernel<<<SDIV(lookup_size, threads_per_block), threads_per_block, 0, stream>>>([=] DEVICEQUALIFIER () {
                        smallsize tid = threadIdx.x + blockIdx.x * blockDim.x;
                        smallsize gid = tid >> group_size_log;
                        smallsize local_tid = tid & (group_size - 1);

                        smallsize offset = gid;
                        smallsize agg = 0;
                        for (smallsize it = 0; it < 1024; ++it) {
                            offset = 1919191919u * offset + 555555555u;
                            auto actual_offset = (offset + local_tid) & (build_size - 1);
                            agg += elements_ptr[actual_offset];
                        }
                        output_elements_ptr[tid] = agg;
                    });
                }
            }
            rc::auto_commit_result(rc_coalesced)
                .add_parameter("key_size", sizeof(element_type))
                .add_parameter("threads_per_block_log", threads_per_block_log)
                .add_parameter("group_size_log", group_size_log)
                .add_measurement("time_ms", time_ms / runs);
        }
    }
    rc_sort.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);
    rc_coalesced.write_csv(std::cout, rc::first_line_header, rc::wide_form, rc::pad_columns);
    cudaStreamDestroy(stream);
}


//#define REBUILD_ON
template <typename key_type>
void generate_keys_hybrid_skip_readin(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::ifstream infile(filename);
    if (infile) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
        }
        infile.close();
        std::cerr << "Loaded " << generated_keys.size() << " keys from cache file: " << filename << std::endl;
        return; // Use existing keys if found
    }

    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys;
    std::cerr << "Generating Hybrid keys..." << std::endl;

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }
    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;
    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 11;
    key_type skip_value = 111;
    key_type skip_jump = 100000;
    key_type new_key = 0;

    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        new_key = current_key;

        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
            new_key += skip_value;
        }
        std::cerr << "Skip-Pattern Next Insert Batch " << skip_pattern_vector.size() << std::endl;
        start_value = new_key + skip_jump;
    }
    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());
    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

// --------------------------------------------------------------

template <typename key_type>
void generate_keys_hybrid_skip(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys; // Ensure uniqueness

    std::cerr << "Generating Hybrid keys..." << std::endl;

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

// -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;

    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 11; // First key
    key_type skip_value = 111; // Jump after each batch
    key_type skip_jump = 100000; // Jump after each batch
    key_type new_key =0;

    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        new_key = current_key;

        // Generate batch of `insert_list_size` keys
        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break; // Stop if we reach required size
            //new_key = current_key; // Skip pattern
            if (unique_keys.insert(new_key).second) { //make sure not a duplicate from uniform_vector
                skip_pattern_vector.push_back(new_key);
            }
            new_key = new_key+ skip_value; // + i; // Skip pattern

        }
        std::cerr << "Skip-Pattern Next Insert Batch " << skip_pattern_vector.size() << std::endl;
        // Move to next batch
        start_value = new_key+ skip_jump;
    }

    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
            std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

// Generate hybrid-dense keys using uniform + dense pattern strategy and save to text file

template <typename key_type>
void generate_keys_hybrid_dense(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    bool shifting_dense_keys = false, // Set to true to generate dense keys
    const std::string& filename = "keys_cache.txt",
    int percentage_distribution_receives_keys = 10 // defaults to 10%
) {
    std::mt19937_64 gen(42);
    std::unordered_set<key_type> unique_keys;

    std::cerr << "Generating Hybrid-Dense keys with " << build_size << " for build in the uniform set and "
              << (size - build_size) << " in the dense pattern..." << std::endl;

    //  size_t uniform_size = static_cast<size_t>(size * uniform_ratio);
    size_t dense_size = size;

    //--------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniform keys..." << std::endl;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    //--------------------- Step 2: Generate Dense Pattern Keys --------------------
    std::cerr << "Generating Dense-Pattern keys..." << std::endl;
    std::vector<key_type> dense_pattern_vector;
    key_type range = static_cast<key_type>((max_usable_key - min_usable_key) * (percentage_distribution_receives_keys / 100.0));
    key_type top_range_key = min_usable_key + range;
    std::uniform_int_distribution<key_type> dense_dist(min_usable_key, top_range_key);

    while (dense_pattern_vector.size() < dense_size) {
        key_type key = dense_dist(gen);
        if (unique_keys.insert(key).second) { // .second method returns true if key was inserted
            dense_pattern_vector.push_back(key);
        }
    }

    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), dense_pattern_vector.begin(), dense_pattern_vector.end());

    std::cerr << "Hybrid-Dense key generation completed." << std::endl;

    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}


template <typename key_type>
void generate_keys_hybrid_dense_file(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    bool shift_dense_keys = false, // Set to true to generate dense keys 
    size_t percentage_distribution_dense_keys = 10,
    size_t percentage_new_keys_from_dense_region =75, // defaults to 75
    const std::string& filename = "keys_cache.txt"
   
) {
    namespace fs = std::filesystem;
    std::string full_path = "data_cache/" + filename;

    if (fs::exists(full_path)) {
        std::ifstream infile(full_path);
        if (infile) {
            generated_keys.clear();
            key_type key;
            while (generated_keys.size() < size && infile >> key) {
                generated_keys.push_back(key);
            }
            if (generated_keys.size() == size) {
                std::cerr << "Loaded keys from cache: " << full_path << std::endl;
                return;
            } else {
                std::cerr << "Cache incomplete, regenerating keys..." << std::endl;
            }
        }
    }

    //--------------------- Step 1: Generate Uniform Keys --------------------
    std::mt19937_64 gen(42);
    std::unordered_set<key_type> unique_keys;

    std::cerr << "Generating Hybrid-Dense keys with " << build_size << " uniform keys and "
              << (size - build_size) << " dense keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    //--------------------- Step 2: Generate Dense Keys --------------------
    std::cerr << "Generating Dense-Pattern keys..." << std::endl;
    std::vector<key_type> dense_pattern_vector;
    key_type range = static_cast<key_type>((max_usable_key - min_usable_key) * (percentage_distribution_dense_keys / 100.0));
    key_type top_range_key = min_usable_key + range;
    std::uniform_int_distribution<key_type> dense_dist(min_usable_key, top_range_key);

    while (dense_pattern_vector.size() < size) {
        key_type key = dense_dist(gen);
        if (unique_keys.insert(key).second) {
            dense_pattern_vector.push_back(key);
        }
    }

    //%% add on the remianing 25% 

    //--------------------- Step 3: MERGE --------------------
    generated_keys.clear();
    generated_keys.reserve(size);
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), dense_pattern_vector.begin(), dense_pattern_vector.end());

    fs::create_directories("data_cache");
    std::ofstream outfile(full_path, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        std::cerr << "Saved generated keys to cache: " << full_path << std::endl;
    } else {
        std::cerr << "Failed to save keys to file: " << full_path << std::endl;
    }
}

template <typename key_type>
void generate_keys_file(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& base_filename
) {
    namespace fs = std::filesystem;

    std::string filename = "data_cache/" + base_filename;
    if (fs::exists(filename)) {
        std::ifstream infile(filename, std::ios::binary);
        if (infile) {
            generated_keys.resize(size);
            infile.read(reinterpret_cast<char*>(generated_keys.data()), sizeof(key_type) * size);
            return;
        }
    }

    std::mt19937_64 gen(42);
    std::uniform_int_distribution<key_type> uniform_key_dist(min_usable_key+2, max_usable_key-3);

    std::unordered_set<key_type> generated_keys_set;
    while (generated_keys_set.size() < size) {
        generated_keys_set.insert(uniform_key_dist(gen));
    }

    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());

    fs::create_directories("data_cache");
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<const char*>(generated_keys.data()), sizeof(key_type) * size);
    }
}
//--------------------------

template <typename key_type>
void generate_keys(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); //ensuring deterministic behaviour
    std::uniform_int_distribution<key_type> uniform_key_dist(min_usable_key, max_usable_key);

    std::unordered_set<key_type> generated_keys_set; //unordered set so keys are unique, not repeating
    while (generated_keys_set.size() < size) {
        generated_keys_set.insert(uniform_key_dist(gen));
    }

    generated_keys.resize(size);
    std::copy(generated_keys_set.begin(), generated_keys_set.end(), generated_keys.begin());

    //randomness and reproducability due to Fixed Keys
}

template <typename key_type>
void draw_probes(
    size_t size,
    const std::vector<key_type>& build_keys,
    size_t active_range_start,
    size_t active_range_end,
    bool start_expected_offset_at_zero,
    std::vector<key_type>& probe_keys,
    std::vector<smallsize>& expected_result
) {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> index_dist(active_range_start, active_range_end - 1);

    //A uniform distribution (std::uniform_int_distribution<size_t> index_dist(active_range_start, active_range_end - 1)) 
    //selects random indices within the active range.

    probe_keys.resize(size);
    expected_result.resize(size);

    for (size_t i = 0; i < size; ++i) {
        size_t index = index_dist(gen);
        probe_keys[i] = build_keys[index];
        expected_result[i] = start_expected_offset_at_zero ? index - active_range_start : index;
    }
    // expected result is the just (offset)index of the key in the active range
}

bool is_all_misses(const std::vector<smallsize>& output, std::string& error_message) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (not_found != output[i]) {
            error_message = "data mismatch at index " + std::to_string(i) + ": expected MISS, but received " + std::to_string(output[i]);
            return false;
        }
    }
    return true;
}

void check_all_misses(const std::vector<smallsize>& output) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (not_found != output[i]) {
            std::cerr << "data mismatch at index " << i << ": expected MISS, but received " << output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}
template <typename key_type>
void check_all_misses_results(const std::vector<smallsize>& output, const std::vector<key_type>& probe_keys) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (not_found != output[i]) {
            std::cerr << "data mismatch at index " << i << ": expected MISS for key " << probe_keys[i] << ", but received " << output[i] << std::endl;
            throw std::logic_error("stop");
        }
    }
}

struct update_configuration {
    std::string description;
    size_t build_size_log;
    size_t probe_size_log;
    size_t batch_count;
    size_t total_inserts_percentage_of_build_size;
    size_t total_deletes_percentage_of_build_size;
   // size_t cache_line;
   // size_t node_size_log;
    bool alternating_insert_delete;
    bool insert_before_delete;
};


template <typename index_type>
void benchmark_updates(
    rc::result_collector& rc,
    size_t runs
) {
    using key_type = typename index_type::key_type;

    size_t free_memory_bytes, total_memory_bytes;
    cudaMemGetInfo(&free_memory_bytes, &total_memory_bytes);

    // todo sync???
    constexpr bool supports_updates = index_type::can_update == operation_support::async;

    std::vector<update_configuration> test_configuration_options {
        //{"delete-insert", 26, 27, 1, 10, 10, false, false},
        //{"insert-delete", 26, 27, 1, 10, 10, false, true},
        //{"batches", 22, 23, 4, 150, 150, false, true},
        //{"batches", 22, 23, 2, 200, 200, false, true},
         //{"batches", 15, 16, 4, 125, 125, false, true},
         //{"batches", 16, 17, 8, 100, 100, false, true},
         // {"batches", 25, 26, 2, 100, 100, false, true},
          {"batches", 15, 16, 8, 200, 200, false, true},
    };

    for (size_t tci = 0; tci < test_configuration_options.size(); ++tci) {
        const auto& tc = test_configuration_options[tci];

        rti_assert(tc.probe_size_log >= tc.build_size_log);

        size_t build_size = size_t{1} << tc.build_size_log;
        size_t key_generation_size = build_size * (100 + tc.total_inserts_percentage_of_build_size) / 100;
        size_t probe_size = size_t{1} << tc.probe_size_log;
        size_t insert_batch_size = build_size * tc.total_inserts_percentage_of_build_size / (100 * tc.batch_count);
        size_t delete_batch_size = build_size * tc.total_deletes_percentage_of_build_size / (100 * tc.batch_count);
        //size_t cache_line = tc.cache_line;
       // size_t node_size_log = size_t{1} << tc.node_size_log;

        rti_assert(key_generation_size * 2 < std::numeric_limits<smallsize>::max());

        std::cerr << "  update experiment " << tc.description << " for " << index_type::short_description() << std::endl;
        std::cerr << "  bits: " << sizeof(key_type) * 8 << std::endl;
        std::cerr << "  build_size_log: " << tc.build_size_log << std::endl;
        std::cerr << "  probe_size_log: " << tc.probe_size_log << std::endl;
        std::cerr << "  batch_count: " << tc.batch_count << std::endl;
        std::cerr << "  alternating_insert_delete: " << tc.alternating_insert_delete << std::endl;
        std::cerr << "  insert_before_delete: " << tc.insert_before_delete << std::endl;

        std::cerr << "  total_inserts_percentage_of_build_size: " << tc.total_inserts_percentage_of_build_size << std::endl;
        std::cerr << "  total_deletes_percentage_of_build_size: " << tc.total_deletes_percentage_of_build_size << std::endl;
        // std::cerr << " cache_line: " << tc.cache_line << std::endl;
        // std::cerr << " node_size_log: " << tc.node_size_log << std::endl;

        std::cerr << "  insert_batch_size: " << insert_batch_size << std::endl;
        std::cerr << "  delete_batch_size: " << delete_batch_size << std::endl;
         std::cerr << "  parameters: " << std::endl;
        for (auto& [key, value] : index_type::parameters()) {
            std::cerr << "    " << key << ": " << value << std::endl;
        }

        /*
        // gen input
        std::vector<key_type> generated_keys;
        generate_keys(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);
        //generate_keys_zipf(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);
        */

#ifdef REGKEYGEN
#pragma message "Regular key generation"
        // -------------------- REGULAR GENERATED KEYS
        
        std::vector<key_type> generated_keys;
        generate_keys_file(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys, "keys_cache.txt");
        //generate_keys(key_generation_size, min_usable_key<key_type>(), max_usable_key<key_type>(), generated_keys);

        std::cerr << "Regular generated key set of size " << key_generation_size << std::endl;

#elif defined(HYBRIDKEYGEN)
#pragma message "Skewed Hybrid Dist key generation"

        // -------------------- NEW GENERATED KEYS
        // Generate input keys
        std::vector<key_type> generated_keys;

        // Call generate_keys_hybrid
        generate_keys_hybrid_skip(
            key_generation_size,  // Total number of keys
            build_size,           // Number of uniform keys
            insert_batch_size,    // Number of insert keys  
            min_usable_key<key_type>(),
            max_usable_key<key_type>(),
            generated_keys
        );

            // Print build_size
        std::cout << "FINISHED HYBRID GEN: Build Size: " << build_size << std::endl;

#elif defined(DENSEKEYGEN)
#pragma message "Hybrid Dense Key Generation"
        // -------------------- HYBRID DENSE KEYS

        // --------------------------------------------- Prior RRun SORT TEST
        //micro_benchmark_search_and_coalesced<uint32_t>();
        // Generate input keys
        std::vector<key_type> generated_keys;
        bool rotate_dense_key_range = false; // Set to true to generate dense keys
       // size_t percentage_distribution_dense_keys = 25; // defaults to 50%
       // size_t percentage_new_keys_from_dense_region = 90; // defaults to 90%
        generate_keys_hybrid_dense_file2_shifted<key_type>(
            key_generation_size,  // Total number of keys
            build_size,                  // 50% uniform keys
            insert_batch_size,    // Number of insert keys  
            tc.batch_count, // Number of batches
            min_usable_key<key_type>(),
            max_usable_key<key_type>(),
            generated_keys,
            shift_insert_range,
            percentage_distribution_dense_keys, // Percentage of keys in the dense pattern (25% of the total key range)
            percentage_new_keys_from_dense_region
        );

        // Print build_size
        std::cout << "FINISHED HYBRID DENSE GEN: Build Size: " << build_size << std::endl;
#endif


#ifdef PRINT_GENERATED_KEYS
#pragma message "PRINT_GENERATED_HYBRID_KEYS=YES"
        // Print first `build_size` uniform keys
        std::cout << "Uniform Keys: " << std::endl;
        for (size_t i = 0; i < build_size; ++i) {
            std::cout << generated_keys[i] << " ";
            if (i % 10 == 9) std::cout << std::endl; // Print 10 keys per line
        }
        std::cout << "\n\n"; // Add gap

        // Print Dense keys
        std::cout << "REST KEYS:" << std::endl;
        for (size_t i = build_size; i < generated_keys.size(); ++i) {
            std::cout << generated_keys[i] << std::endl; // One key per line
        }

        std::cout << std::endl;
#endif
        
        // ---------------------------------------------
        std::cerr << "generated key set of size " << key_generation_size << std::endl;

        cuda_buffer<key_type> insert_delete_keys_buffer;

        #ifdef COMBINE_INSERT_DELETE
        cuda_buffer<key_type> insert_keys_buffer;
        cuda_buffer<key_type> delete_keys_buffer;
        insert_keys_buffer.alloc(insert_batch_size);
        delete_keys_buffer.alloc(delete_batch_size);
        #endif

        cuda_buffer<smallsize> insert_offsets_buffer;
        cuda_buffer<key_type> probe_keys_buffer;
        cuda_buffer<smallsize> result_buffer;
         // not enough memory for setup (with 10% margin)
        size_t setup_bytes = (key_generation_size * sizeof(key_type) + insert_batch_size * sizeof(smallsize) + probe_size * (sizeof(key_type) + sizeof(smallsize))) * 11 / 10;
        if (free_memory_bytes < setup_bytes) {
            std::cerr << " -> skipping, not enough memory for setup: " << setup_bytes << " bytes > " << free_memory_bytes << " bytes" << std::endl;
            continue;
        }
        size_t available_bytes_for_index = free_memory_bytes - setup_bytes;
        insert_delete_keys_buffer.alloc(key_generation_size);
        insert_offsets_buffer.alloc(insert_batch_size);
        probe_keys_buffer.alloc(probe_size);
        result_buffer.alloc(probe_size);
        smallsize next_free=0;



        for (size_t run = 0; run < runs; ++run) {
            cuda_timer timer(0);

            std::cerr << " -> run " << run << std::endl;

            size_t active_range_start = 0;
            size_t active_range_end = build_size;

            // upload initial build keys
            insert_delete_keys_buffer.upload(generated_keys.data(), active_range_end - active_range_start);

            // initial build (untimed)
            index_type index;
           // index.build(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, nullptr, nullptr);
            #ifdef DEBUG_BENCHMARK_OUTPUT
                next_free= index.allocation_buffer_next_free();
                smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                std::cerr << "After Build  # of Assigned Free Nodes : " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
            #endif
            try {
                // todo final size does not have to be equal to key_generation_size, make sure to compute that properly
                // Note:  max size will not always be key_generation_size, it can be smaller if the index does not need all keys
                //index.build(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, key_generation_size, available_bytes_for_index, nullptr, nullptr);
                index.build_static_tree(insert_delete_keys_buffer.ptr(), active_range_end - active_range_start, key_generation_size, available_bytes_for_index, nullptr, nullptr);

          
            } catch (const std::exception& e) {
                std::cerr << "   -> SKIP " << e.what() << std::endl;
                break;
            }



#ifdef COMBINE_INSERT_DELETE
#pragma message "COMBINE_INSERT_DELETE=YES"
            // run updates
            for (size_t step = 0; step < tc.batch_count + 1; ++step) {
#else
#pragma message "COMBINE_INSERT_DELETE=NO"

            for (size_t step = 0; step < 2 * tc.batch_count + 1; ++step) {
#endif

                double insert_or_delete_time_ms = 0;
                double probe_time_ms = 0;
                double rebuild_time_ms = 0;
                double sort_time_ms = 0;
                double index_layer_time_ms = 0;
                double bucket_layer_time_ms = 0;
                size_t before_update_bytes = 0;
                size_t after_update_bytes = 0;

                bool correct = true;
                std::string error_message;
                size_t totalkeys =0;

                bool do_insert =
                    !tc.alternating_insert_delete && tc.insert_before_delete  && (step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete  && tc.insert_before_delete  && ((step & 1) == 1) ||
                    !tc.alternating_insert_delete && !tc.insert_before_delete && !(step < tc.batch_count + 1) ||
                    tc.alternating_insert_delete  && !tc.insert_before_delete && !((step & 1) == 1);
                bool do_delete = !do_insert;

                size_t new_active_range_end = do_insert ? active_range_end + insert_batch_size : active_range_end;
                size_t new_active_range_start = do_delete ? active_range_start + delete_batch_size : active_range_start;

                size_t before_update_size = active_range_end - active_range_start;
                size_t after_update_size = new_active_range_end - new_active_range_start;
                
                if (step == 0) {
                    std::cerr << "    step 0: probe only" << std::endl;
                    goto probe;
                }

                std::cerr << "    step " << step << ": ";
                std::cerr << (do_insert ? "insert" : do_delete ? "delete" : "nothing") << std::endl;

                // active range has to be non-empty
                if (new_active_range_start >= new_active_range_end) {
                    goto end_run;
                }

                before_update_bytes = index.gpu_resident_bytes();

                if constexpr (supports_updates) {
                    
                    // a Dummy round of Insertions:
                    if (step == 1 && index.name == "cg_rtx_index_updates") {
                        printf("performing dummy inserts\n");
                        size_t dummy_insert_size = 10; //2000;
                        if (dummy_insert_size > active_range_end - active_range_start) {
                            dummy_insert_size = active_range_end - active_range_start; // Ensure we don't exceed available keys
                        }
                        std::vector<key_type> dummy_inserts(
                            generated_keys.begin(),
                            generated_keys.begin() + dummy_insert_size
                        );

                        std::vector<smallsize> dummy_offsets(dummy_insert_size);
                        std::iota(dummy_offsets.begin(), dummy_offsets.end(), 0);

                        auto sort_perm_dum = sort_permutation(dummy_inserts, std::less<key_type>());
                        apply_permutation(dummy_inserts, sort_perm_dum);
                        apply_permutation(dummy_offsets, sort_perm_dum);

                        insert_delete_keys_buffer.upload(dummy_inserts.data(), dummy_insert_size);
                        insert_offsets_buffer.upload(dummy_offsets.data(), dummy_insert_size);

                        std::cerr << "Dummy insert size: " << dummy_insert_size << std::endl;
                        //NOTE Should be SORTING the DUMMY LISTS but it does not matter All keys already exist
                        // timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), dummy_insert_size, 0);
                        //index.insert(nullptr, nullptr, dummy_insert_size, 0);
                        //timer.stop();
                        cudaDeviceSynchronize(); CUERR
                    }


#ifdef COMBINE_INSERT_DELETE
                // insert the next batch
                    if (do_insert) {
                        std::vector<key_type> relevant_inserts(generated_keys.begin() + active_range_end, generated_keys.begin() + active_range_end + insert_batch_size);
                        std::vector<smallsize> associated_offsets(insert_batch_size);
                        std::iota(associated_offsets.begin(), associated_offsets.end(), active_range_end);
                        auto sort_perm = sort_permutation(relevant_inserts, std::less<key_type>());
                        apply_permutation(relevant_inserts, sort_perm);
                        apply_permutation(associated_offsets, sort_perm);
                        insert_keys_buffer.upload(relevant_inserts.data(), insert_batch_size);
                        insert_offsets_buffer.upload(associated_offsets.data(), insert_batch_size);
                        C2EX
                        std::cerr << "insert size: " << insert_batch_size << std::endl;
                        /*
                        timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                         #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Insert  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                        */
                    }
                     // or delete the next batch
                    if (do_delete) {
                        std::vector<key_type> relevant_deletes(generated_keys.begin() + active_range_start, generated_keys.begin() + active_range_start + delete_batch_size);
                        std::sort(relevant_deletes.begin(), relevant_deletes.end());
                        delete_keys_buffer.upload(relevant_deletes.data(), delete_batch_size);
                        C2EX
                        std::cerr << "delete size: " << delete_batch_size << std::endl;
                        /*
                        timer.start();
                        index.remove(insert_delete_keys_buffer.ptr(), delete_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                        #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Delete  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                        */
                    }
                        timer.start();
                        index.insert_and_remove(insert_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, delete_keys_buffer.ptr() ,0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                         #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Combined  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif


#else

                    // insert the next batch
                    if (do_insert) {

                        std::vector<key_type> relevant_inserts(generated_keys.begin() + active_range_end, generated_keys.begin() + active_range_end + insert_batch_size);
                        std::vector<smallsize> associated_offsets(insert_batch_size);
                        std::iota(associated_offsets.begin(), associated_offsets.end(), active_range_end);
                        auto sort_perm = sort_permutation(relevant_inserts, std::less<key_type>());
                        apply_permutation(relevant_inserts, sort_perm);
                        apply_permutation(associated_offsets, sort_perm);
                        insert_delete_keys_buffer.upload(relevant_inserts.data(), insert_batch_size);
                        insert_offsets_buffer.upload(associated_offsets.data(), insert_batch_size);
                        C2EX
                        std::cerr << "insert size: " << insert_batch_size << std::endl;
                        timer.start();
                        index.insert(insert_delete_keys_buffer.ptr(), insert_offsets_buffer.ptr(), insert_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                         #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Insert  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                    }

                    // or delete the next batch
                    if (do_delete) {
                        std::vector<key_type> relevant_deletes(generated_keys.begin() + active_range_start, generated_keys.begin() + active_range_start + delete_batch_size);
                        std::sort(relevant_deletes.begin(), relevant_deletes.end());
                        insert_delete_keys_buffer.upload(relevant_deletes.data(), delete_batch_size);
                        C2EX
                        std::cerr << "delete size: " << delete_batch_size << std::endl;
                        timer.start();
                        index.remove(insert_delete_keys_buffer.ptr(), delete_batch_size, 0);
                        timer.stop();
                        insert_or_delete_time_ms = timer.time_ms();
                        cudaDeviceSynchronize(); CUERR
                        #ifdef DEBUG_BENCHMARK_OUTPUT
                            next_free= index.allocation_buffer_next_free();
                            //smallsize total_alloc_nodes= index.allocation_buffer_total_nodes();
                             std::cerr << "After Delete  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;
                        #endif
                    }

#endif
                }

                // adjust active range
                active_range_end = new_active_range_end;
                active_range_start = new_active_range_start;

                 if constexpr (!supports_updates) {
                    index.destroy();
                    size_t rebuild_size = active_range_end - active_range_start;
                    insert_delete_keys_buffer.upload(generated_keys.data() + active_range_start, rebuild_size);
                    try {
                        index.build(insert_delete_keys_buffer.ptr(), rebuild_size, rebuild_size, available_bytes_for_index, &insert_or_delete_time_ms, nullptr);
                    } catch (const std::exception& e) {
                        std::cerr << " -> SKIP " << e.what() << std::endl;
                        break;
                    }
                }

                std::cerr << "     -> " << insert_or_delete_time_ms << " ms" << std::endl;

probe:;
                // probe all inserted keys
                if (active_range_end > active_range_start) {
                    std::vector<key_type> probe_keys;
                    std::vector<smallsize> expected_result;
                    draw_probes(probe_size, generated_keys, active_range_start, active_range_end, !supports_updates, probe_keys, expected_result);
                    probe_keys_buffer.upload(probe_keys.data(), probe_size);
                    result_buffer.zero();

                    const bool sort_probes = true;
                    if (sort_probes) {
                        // --- sort on device (keys-only) ---
                        cuda_buffer<key_type>    d_sorted_keys;
                        cuda_buffer<uint32_t>    d_perm_sorted;
                        cuda_buffer<uint8_t>     d_aux;
                        d_sorted_keys.alloc(probe_size);
                        d_perm_sorted.alloc(probe_size);
                        d_aux.alloc(probe_size * sizeof(key_type) / sizeof(uint8_t) + 1024);

                        key_only_sort_device<key_type>(probe_keys_buffer, probe_size,
                                                    d_sorted_keys, d_perm_sorted, d_aux, /*stream*/0);

                        // --- run lookups on sorted keys ---
                        std::cerr << "      Index Layer Lookup (sorted) ";
                        timer.start();
                        index.index_layer_lookups(d_sorted_keys.ptr(), probe_size, 0);
                        timer.stop();
                        index_layer_time_ms = timer.time_ms();
                        std::cerr << index_layer_time_ms << " ms - ";

                        cuda_buffer<smallsize> tmp_results_sorted;
                        tmp_results_sorted.alloc(probe_size);
                        tmp_results_sorted.zero();

                        std::cerr << "Bucket Layer Lookup (sorted) ";
                        timer.start();
                        index.bucket_layer_lookups(d_sorted_keys.ptr(),
                                                tmp_results_sorted.ptr(),
                                                probe_size, 0);
                        timer.stop();
                        bucket_layer_time_ms = timer.time_ms();
                        std::cerr << bucket_layer_time_ms << " ms ---> "
                                << "  Total: " << (bucket_layer_time_ms + index_layer_time_ms) << " ms" << std::endl;

                        // --- scatter back to original order expected by checks ---
                        scatter_sorted_to_original<smallsize>(tmp_results_sorted,
                                                            d_perm_sorted,
                                                            result_buffer,
                                                            probe_size, 0);
                    } else {
                        std::cerr << "      Index Layer Lookup ";
                        timer.start();
                        index.index_layer_lookups(probe_keys_buffer.ptr(), probe_size, 0);
                        timer.stop();
                        index_layer_time_ms = timer.time_ms();
                        std::cerr << index_layer_time_ms << " ms - ";

                        std::cerr << "Bucket Layer Lookup ";
                        timer.start();
                        index.bucket_layer_lookups(probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, 0);
                        timer.stop();
                        bucket_layer_time_ms = timer.time_ms();
                        std::cerr << bucket_layer_time_ms << " ms ---> "
                                << "  Total: " << (bucket_layer_time_ms + index_layer_time_ms) << " ms" << std::endl;
                    }





                  
                    /* 
                    std::cerr << "Probe Keys:  ";
                    timer.start();
                     // index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, 0);
                     index.lookup_static_tree(probe_keys_buffer.ptr(), result_buffer.ptr(), probe_size, 0);
                    timer.stop();

                    */

                   // printf("Lookup Kernel Time: %.3f ms\n", timer.time_ms());
                    probe_time_ms = timer.time_ms();
                    cudaDeviceSynchronize(); C2EX

                   // check_result(probe_keys, probe_keys, expected_result, result_buffer);
                    correct = is_matching_result(probe_keys, probe_keys, expected_result, result_buffer, error_message);
                    if (!correct) {
                        std::cerr << " -> SKIP " << error_message << std::endl;
                        break;
                    }
                }

                // check if keys before the inserted range are all misses
               /*  if (active_range_start > 0) {
                    probe_keys_buffer.upload(generated_keys.data(), active_range_start);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), active_range_start - 0, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(active_range_start - 0);
                    check_all_misses(result);
                }
                */

                /*
                // check if keys beyond the inserted range are all misses
                if (active_range_end < key_generation_size) {
                    probe_keys_buffer.upload(generated_keys.data() + active_range_end, key_generation_size - active_range_end);
                    result_buffer.zero();

                    index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), key_generation_size - active_range_end, 0);
                    cudaDeviceSynchronize();

                    auto result = result_buffer.download(key_generation_size - active_range_end);
                    check_all_misses(result);
                }
                */
               

                // check if keys before the inserted range are all misses
                if (active_range_start > 0) {
                    probe_keys_buffer.upload(generated_keys.data(), active_range_start);
                    result_buffer.zero();
                     index.lookup_static_tree(probe_keys_buffer.ptr(), result_buffer.ptr(), active_range_start, 0);

                     //index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), active_range_start, 0);
                    cudaDeviceSynchronize();
                    //printf("Lookup Kernel Misses \n");


                    auto result = result_buffer.download(active_range_start);
                    std::vector<key_type> probe_keys(generated_keys.begin(), generated_keys.begin() + active_range_start);
                    check_all_misses_results<key_type>(result, probe_keys);
                    correct = is_all_misses(result, error_message);
                    if (!correct) {
                        std::cerr << " -> SKIP " << error_message << std::endl;
                        break;
                    }
                        
                    
                }

                // check if keys beyond the inserted range are all misses
                if (active_range_end < key_generation_size) {
                    probe_keys_buffer.upload(generated_keys.data() + active_range_end, key_generation_size - active_range_end);
                    result_buffer.zero();

                    index.lookup_static_tree(probe_keys_buffer.ptr(), result_buffer.ptr(), key_generation_size - active_range_end, 0);
                    //index.lookup(probe_keys_buffer.ptr(), result_buffer.ptr(), key_generation_size - active_range_end, 0); 
                    cudaDeviceSynchronize();

                    //printf("Lookup Kernel Misses 2 \n");

                    auto result = result_buffer.download(key_generation_size - active_range_end);
                    std::vector<key_type> probe_keys(generated_keys.begin() + active_range_end, generated_keys.end());
                    check_all_misses_results<key_type>(result, probe_keys);
                    correct = is_all_misses(result, error_message);
                    if (!correct) {
                        std::cerr << " -> SKIP " << error_message << std::endl;
                        break;
                    }

                }

                // Here 
                //                                std::cerr << "     -> " << probe_time_ms << " ms" << std::endl;

                after_update_bytes = index.gpu_resident_bytes();

#ifdef COMPUTE_TOTALKEYS
#pragma message "COMPUTE_TOTALKEYS=YES"
                totalkeys = index.compute_total_size(0);
                std::cerr << "Total Keys in DS: " << totalkeys <<  std::endl;


#else
#pragma message "COMPUTE_TOTALKEYS=NO"
                //std::cerr << "Total Keys in DS: " << after_update_size <<  std::endl;
#endif
 
#ifdef REBUILD_ON
#pragma message "REBUILD_ON=YES"

                // rebuild 
            rebuild_time_ms = 0;
            if (step % rebuild_frequency == 0) {
                next_free = index.allocation_buffer_next_free();
                timer.start();
                index.rebuild_static_tree(nullptr, nullptr);
                timer.stop();
                rebuild_time_ms = timer.time_ms();
                //next_free = index.allocation_buffer_next_free();
                std::cerr << "Rebuild ON:  # Used new nodes prior to Rebuild: " << next_free << " Rebuild time: " << rebuild_time_ms << std::endl;

            }       
#endif 
#ifdef DEBUG_BENCHMARK_OUTPUT
              next_free= index.allocation_buffer_next_free();
              total_alloc_nodes= index.allocation_buffer_total_nodes();
#ifdef REBUILD_ON
                std::cerr << "After- Rebuild ON  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << " Rebuild time: " << rebuild_time_ms << std::endl;
#else
                std::cerr << "After- NO Rebuild  #of Used New Nodes: " << next_free<<  " of total available nodes: " << total_alloc_nodes << std::endl;   
#endif
#endif  
            {    
            auto index_id = index_type::short_description() + "__" + parameters_to_name(index_type::parameters());
                rc::auto_commit_result(rc)
                    .add_parameter("EXPERIMENT", tc.description)
                    .add_parameter("index_type", index_type::short_description())
                    .add_parameter("index_parameters", parameters_to_string(index_type::parameters()))
                    .add_parameter("index_id", index_id)
                    .add_parameter("run", run)
                    .add_parameter("key_bits", sizeof(key_type) * 8)
                    .add_parameter("build_size_log", tc.build_size_log)
                    .add_parameter("probe_size_log", tc.probe_size_log)
                    .add_parameter("batch_count", tc.batch_count)
                    .add_parameter("total_inserts_percentage_of_build_size", tc.total_inserts_percentage_of_build_size)
                    .add_parameter("total_deletes_percentage_of_build_size", tc.total_deletes_percentage_of_build_size)
                    //.add_parameter("cache_line_size", tc.cache_line)
                   // .add_parameter("node_size_log", tc.node_size_log)
                    .add_parameter("alternating_insert_delete", tc.alternating_insert_delete)
                    .add_parameter("insert_before_delete", tc.insert_before_delete)
                    .add_parameter("step", step)
                    .add_parameter("do_insert", do_insert)
                    .add_parameter("do_delete", do_delete)
                    .add_parameter("current_batch_size", do_insert ? insert_batch_size : do_delete ? delete_batch_size : 0)
                    .add_parameter("before_update_size", before_update_size)
                    .add_parameter("after_update_size", after_update_size)
                    .add_measurement("insert_or_delete_time_ms", insert_or_delete_time_ms)
                    .add_measurement("probe_time_ms", probe_time_ms)
                    .add_measurement("sort_time_ms", sort_time_ms)
                    .add_measurement("index_layer_time_ms", index_layer_time_ms)
                    .add_measurement("bucket_layer_time_ms", bucket_layer_time_ms)
                    .add_measurement("rebuild_time_ms", rebuild_time_ms)
                    .add_measurement("before_update_bytes", before_update_bytes)
                    .add_measurement("after_update_bytes", after_update_bytes)
                    ;
            }
end_run:;
            }
        }
    }
}

#endif



/*




template <typename key_type>
void generate_keys_hybrid_zifp(
    size_t size,
    size_t build_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility

    size_t range = max_usable_key - min_usable_key + 1;

    // -------------------- Step 1: Generate Zipfian Keys Using `zipf_index_distribution` --------------------
    std::unordered_set<key_type> zipf_keys;
    std::vector<key_type> zipf_vector;

    std::cerr << "Generating Zipfian-distributed keys (Fast Version)..." << std::endl;

    double zipfian_s = 0.5;
    zipf_index_distribution zipf_dist(size - build_size, zipfian_s); // Using the fast Zipf distribution

    while (zipf_keys.size() < size - build_size) {
        key_type key = min_usable_key + zipf_dist(gen);  // Faster Zipfian sampling
        if (zipf_keys.insert(key).second) {
            zipf_vector.push_back(key);
        }
    }

    std::cerr << "Zipfian keys generated: " << zipf_keys.size() << std::endl;

    // -------------------- Step 2: Generate Uniform Keys --------------------
    std::unordered_set<key_type> uniform_keys;
    std::vector<key_type> uniform_vector;

    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);

    while (uniform_keys.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (zipf_keys.find(key) == zipf_keys.end() && uniform_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_keys.size() << std::endl;

    // -------------------- Step 3: Merge Uniform and Zipfian Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), zipf_vector.begin(), zipf_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 4: Save to File --------------------
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}


template <typename key_type>
void generate_keys_zipf(
    size_t size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "zipf_keys_cache.txt"
) {
    std::ifstream infile(filename);
    if (infile.good()) {
        key_type key;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) break;
        }
        infile.close();
        if (generated_keys.size() == size) {
            std::cerr << "Loaded keys from cache: " << filename << std::endl;
            return;
        }
    }

    std::cerr << "Generating new Zipfian keys..." << std::endl;
    std::mt19937_64 gen(42); // Ensuring deterministic behavior
    size_t range = max_usable_key - min_usable_key + 1;

    // Compute Zipfian probabilities
    double zipfian_s = 0.5;
    double sum = 0.0;
    std::vector<double> probabilities(range);
    for (size_t i = 1; i <= range; ++i) {
        sum += 1.0 / std::pow(i, zipfian_s);
    }
    for (size_t i = 1; i <= range; ++i) {
        probabilities[i - 1] = (1.0 / std::pow(i, zipfian_s)) / sum;
    }

    std::discrete_distribution<size_t> zipf_dist(probabilities.begin(), probabilities.end());

    std::unordered_set<key_type> unique_keys;
    while (unique_keys.size() < size) {
        key_type key = min_usable_key + zipf_dist(gen);
        unique_keys.insert(key);
    }

    // Move unique keys to the output vector
    generated_keys.assign(unique_keys.begin(), unique_keys.end());

    // Write keys to file for future use
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }




template <typename key_type>
void generate_keys_hybrid_skip_readin(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys,
    const std::string& filename = "keys_cache.txt"
) {
    std::ifstream infile(filename);
    generated_keys.clear();
    key_type key;

    if (infile) {
        std::cerr << "Reading keys from cache..." << std::endl;
        while (infile >> key) {
            generated_keys.push_back(key);
            if (generated_keys.size() >= size) {
                std::cerr << "Sufficient keys loaded from cache." << std::endl;
                return;
            }
        }
        infile.close();
    }
    
    std::cerr << "Generating Hybrid keys..." << std::endl;
    std::mt19937_64 gen(42); // Fixed seed for reproducibility
    std::unordered_set<key_type> unique_keys(generated_keys.begin(), generated_keys.end());

    // -------------------- Step 1: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);
    std::vector<key_type> uniform_vector;
    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }
    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 2: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;
    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 111;
    key_type skip_jump = 100000;
    while (unique_keys.size() < size) {
        key_type current_key = start_value;
        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            key_type new_key = current_key + i;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
        }
        start_value += skip_jump;
    }
    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 3: Merge and Save Keys --------------------
    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());
    std::cerr << "Hybrid key generation completed." << std::endl;
    
    std::ofstream outfile(filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}


template <typename key_type>
void generate_keys_hybrid_skip_file(
    size_t size,
    size_t build_size,
    size_t insert_list_size,
    key_type min_usable_key,
    key_type max_usable_key,
    std::vector<key_type>& generated_keys
) {
    std::mt19937_64 gen(42); // Fixed seed for reproducibility

    // Define the cache file path
    std::string cache_filename = "data_cache/keys_cache_" + std::to_string(build_size) + ".txt";

    // -------------------- Step 1: Check for Existing Cache --------------------
    if (std::filesystem::exists(cache_filename)) {
        std::ifstream infile(cache_filename);
        if (infile.good()) {
            key_type key;
            while (infile >> key) {
                generated_keys.push_back(key);
                if (generated_keys.size() >= size) break;
            }
            infile.close();

            if (generated_keys.size() == size) {
                std::cerr << "Loaded keys from cache: " << cache_filename << std::endl;
                return;
            }
        }
    }

    // -------------------- Step 2: Generate Uniform Keys --------------------
    std::cerr << "Generating Uniformly-distributed keys..." << std::endl;

    std::unordered_set<key_type> unique_keys;
    std::vector<key_type> uniform_vector;
    std::uniform_int_distribution<key_type> uniform_dist(min_usable_key, max_usable_key);

    while (uniform_vector.size() < build_size) {
        key_type key = uniform_dist(gen);
        if (unique_keys.insert(key).second) {
            uniform_vector.push_back(key);
        }
    }

    std::cerr << "Uniform keys generated: " << uniform_vector.size() << std::endl;

    // -------------------- Step 3: Generate Skip Pattern Keys --------------------
    std::cerr << "Generating Skip-Pattern keys..." << std::endl;

    std::vector<key_type> skip_pattern_vector;
    key_type start_value = 111; // First key
    key_type skip_jump = 100000; // Jump after each batch

    while (unique_keys.size() < size) {
        key_type current_key = start_value;

        for (size_t i = 0; i < insert_list_size; ++i) {
            if (unique_keys.size() >= size) break;
            key_type new_key = current_key + i;
            if (unique_keys.insert(new_key).second) {
                skip_pattern_vector.push_back(new_key);
            }
        }

        start_value += skip_jump; // Move to next batch
    }

    std::cerr << "Skip-Pattern keys generated: " << skip_pattern_vector.size() << std::endl;

    // -------------------- Step 4: Merge Uniform and Skip Keys --------------------
    generated_keys.clear();
    generated_keys.reserve(size);

    generated_keys.insert(generated_keys.end(), uniform_vector.begin(), uniform_vector.end());
    generated_keys.insert(generated_keys.end(), skip_pattern_vector.begin(), skip_pattern_vector.end());

    std::cerr << "Hybrid key generation completed." << std::endl;

    // -------------------- Step 5: Save to File --------------------
    std::ofstream outfile(cache_filename, std::ios::trunc);
    if (outfile) {
        for (const auto& key : generated_keys) {
            outfile << key << "\n";
        }
        outfile.close();
        std::cerr << "Saved generated keys to cache: " << cache_filename << std::endl;
    } else {
        std::cerr << "Failed to save keys to file!" << std::endl;
    }
}

}*/