#ifndef TREE_H
#define TREE_H

#include "gpu_btree.h"

#include <nvtx3/nvtx3.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using GpuBTree::gpu_blink_tree;

// for nvtx
struct nvtx_b_tree_domain{ static constexpr char const* name{"b_tree"}; };


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_find_kernel(
    const key_type* keys,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
  auto thread_id  = threadIdx.x + blockIdx.x * blockDim.x;

  auto block = cooperative_groups::this_thread_block();
  auto tile  = cooperative_groups::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key     = btree::invalid_key;
  auto value   = btree::invalid_value;
  bool to_find = false;
  if (thread_id < keys_count) {
    key     = keys[thread_id];
    to_find = true;
  }

  using allocator_type = device_allocator_context<typename btree::allocator_type>;
  allocator_type allocator{tree.allocator_, tile};

  auto work_queue = tile.ballot(to_find);
  while (work_queue) {
    auto cur_rank = __ffs(work_queue) - 1;
    auto cur_key  = tile.shfl(key, cur_rank);
    smallsize cur_result;
    cur_result = tree.cooperative_find(cur_key, tile, allocator, concurrent);
    if (cur_rank == tile.thread_rank()) {
      value   = cur_result;
      to_find = false;
    }
    work_queue = tile.ballot(to_find);
  }

  if (thread_id < keys_count) {
    results[thread_id] = value != std::numeric_limits<uint32_t>::max() ? value : not_found;
  }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_range_lookup_kernel(
    const key_type* lower_bounds,
    const key_type* upper_bounds,
    smallsize* results,
    const size_type keys_count,
    btree tree,
    bool concurrent = false
) {
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    auto block                = cooperative_groups::this_thread_block();
    auto tile                 = cooperative_groups::tiled_partition<btree::branching_factor>(block);
    auto tile_id              = thread_id / btree::branching_factor;
    auto first_tile_thread_id = tile_id * btree::branching_factor;

    if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

    auto lower_bound = btree::invalid_key;
    auto upper_bound = btree::invalid_key;

    bool to_find    = false;
    if (thread_id < keys_count) {
      lower_bound = lower_bounds[thread_id];
      upper_bound = upper_bounds[thread_id];
      to_find     = true;
    }

    using allocator_type = device_allocator_context<typename btree::allocator_type>;
    allocator_type allocator{tree.allocator_, tile};

    smallsize result = 0;
    auto work_queue = tile.ballot(to_find);
    while (work_queue) {
      smallsize cur_result = 0;
      auto cur_rank        = __ffs(work_queue) - 1;
      auto cur_lower_bound = tile.shfl(lower_bound, cur_rank);
      auto cur_upper_bound = tile.shfl(upper_bound, cur_rank);
      auto local_result = tree.modified_aggregating_cooperative_range_query(
        cur_lower_bound,
        cur_upper_bound,
        tile,
        allocator,
        concurrent);

      // todo test this
      cur_result = cooperative_groups::reduce(tile, local_result, cooperative_groups::plus<smallsize>());

      if (cur_rank == tile.thread_rank()) {
        result = cur_result;
        to_find = false;
      }
      work_queue = tile.ballot(to_find);
    }

    if (thread_id < keys_count) {
        results[thread_id] = result;
    }
}


template <typename key_type, typename size_type, typename btree>
GLOBALQUALIFIER
void modified_insert_kernel(
    const key_type* keys,
    const size_type keys_count,
    btree tree
) {
  auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  auto block     = cooperative_groups::this_thread_block();
  auto tile      = cooperative_groups::tiled_partition<btree::branching_factor>(block);

  if ((thread_id - tile.thread_rank()) >= keys_count) { return; }

  auto key       = btree::invalid_key;
  auto value     = btree::invalid_value;
  bool to_insert = false;
  if (thread_id < keys_count) {
    key       = keys[thread_id];
    value     = thread_id;
    to_insert = true;
  }
  using allocator_type = typename btree::device_allocator_context_type;
  allocator_type allocator{tree.allocator_, tile};

  size_type num_inserted = 1;
  auto work_queue        = tile.ballot(to_insert);
  while (work_queue) {
    auto cur_rank  = __ffs(work_queue) - 1;
    auto cur_key   = tile.shfl(key, cur_rank);
    auto cur_value = tile.shfl(value, cur_rank);

    tree.cooperative_insert(cur_key, cur_value, tile, allocator);

    if (tile.thread_rank() == cur_rank) { to_insert = false; }
    num_inserted++;
    work_queue = tile.ballot(to_insert);
  }
}


/*
inline void test_modified_btree() {
    std::vector<uint32_t> keys {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<uint32_t> offsets(keys.size());
    std::vector<uint32_t> probes       {1, 4, 5, 2, 1, 3, 7, 10};
    std::vector<uint32_t> probes_upper {5, 5, 5, 2, 1, 3, 8, 20};
    std::vector<uint32_t> results(probes.size());
    cuda_buffer<uint32_t> kb, ob, pb, pub, rb;
    std::iota(offsets.begin(), offsets.end(), 0);
    kb.alloc_and_upload(keys);
    ob.alloc_and_upload(offsets);
    pb.alloc_and_upload(probes);
    pub.alloc_and_upload(probes_upper);
    rb.alloc_and_upload(results);

    using btree = gpu_blink_tree<uint32_t, uint32_t>;

    btree tree;
    tree.bulk_load(kb, ob, keys.size(), true);
    cudaDeviceSynchronize(); CUERR
    //modified_insert_kernel<<<SDIV(keys.size(), 512), 512>>>(kb, keys.size(), tree);
    //modified_find_kernel<<<SDIV(probes.size(), 512), 512>>>(pb, rb, probes.size(), tree);
    modified_range_lookup_kernel<<<SDIV(probes.size(), 512), 512>>>(pb, pub, rb, probes.size(), tree);
    cudaDeviceSynchronize(); CUERR

    rb.download_and_output(results.size());
}
*/


template <typename key_type_, size_t additional_inserts_percent = 0>
class tree {
public:
    using key_type = key_type_;

private:
    using tree_type = gpu_blink_tree<key_type, smallsize>;
    std::optional<tree_type> wrapped_tree;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = false;
    static constexpr bool can_range_lookup = true;
    static constexpr bool can_update = true;

    static std::string short_description() {
        return std::string("b_link_tree_16");
    }

    static size_t estimate_build_bytes(size_t initial_size) {
        size_t final_size = initial_size * (100 + additional_inserts_percent) / 100;
        size_t max_tree_size = std::max(2 * initial_size, final_size);
        size_t sort_bytes = (sizeof(smallsize) + sizeof(key_type)) * initial_size;
        size_t sort_aux_bytes = sizeof(smallsize) * initial_size + find_pair_sort_buffer_size<key_type, smallsize>(initial_size);
        size_t tree_bytes = (sizeof(smallsize) + sizeof(key_type)) * max_tree_size * 12 / 10;
        return std::max(sort_bytes + sort_aux_bytes, sort_bytes + tree_bytes);
    }

    size_t gpu_resident_bytes() {
        return wrapped_tree.value().compute_memory_usage_bytes();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {

        cuda_buffer<key_type> sorted_keys_buffer;
        cuda_buffer<smallsize> sorted_offsets_buffer;

        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;
            sorted_keys_buffer.alloc(size);
            offsets_buffer.alloc(size);
            sorted_offsets_buffer.alloc(size);
            init_offsets(offsets_buffer, size, build_time_ms);

            cudaDeviceSynchronize(); CUERR

            size_t temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes);
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            if (build_bytes) *build_bytes += sorted_keys_buffer.size_in_bytes() + sorted_offsets_buffer.size_in_bytes() + temp_buffer.size_in_bytes() + offsets_buffer.size_in_bytes();

            cudaDeviceSynchronize(); CUERR
        }

        size_t expected_updates = size * additional_inserts_percent / 100;
        wrapped_tree.emplace(size, expected_updates);

        cuda_timer timer(0);
        timer.start();

        wrapped_tree.value().bulk_load(sorted_keys_buffer, sorted_offsets_buffer, size, true, 0);
        //modified_insert_kernel<<<SDIV(size, 512), 512>>>(keys, size, wrapped_tree.value());

        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        nvtx3::scoped_range_in<nvtx_b_tree_domain> launch{"launch"};
        modified_find_kernel<<<SDIV(size, 512), 512, 0, stream>>>(keys, result, size, wrapped_tree.value());
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {}

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {
        modified_range_lookup_kernel<<<SDIV(size, 512), 512, 0, stream>>>(lower, upper, result, size, wrapped_tree.value());
    }

    void destroy() {
        wrapped_tree.reset();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {
        wrapped_tree.value().insert(update_list, offsets, size, stream);
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {
        wrapped_tree.value().erase(update_list, size, stream);
    }
};

#endif
