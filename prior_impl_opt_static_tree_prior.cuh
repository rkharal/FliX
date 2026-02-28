#ifndef IMPL_OPT_STATIC_TREE_CUH
#define IMPL_OPT_STATIC_TREE_CUH

#include "definitions_opt.cuh"
#include "memory_layout.cuh"
#include "utilities.cuh"

#include <cmath>

#include <cub/cub.cuh>

#include <fstream>
#include <vector>
#include <cstring>

namespace mem = memory_layout;

// for nvtx
struct nvtx_opt_static_tree_domain{ static constexpr char const* name{"opt_static_tree"}; };


struct tree_metadata {
    constexpr static smallsize max_level_count = 14;

    smallsize level_count;
    smallsize total_nodes;
    // level 0 is the topmost level
    std::array<smallsize, max_level_count> nodes_on_level;
};


template <typename key_type, uint16_t node_size_log, bool use_row_layout>
GLOBALQUALIFIER
void build_static_tree_kernel(
        mem::store_type<key_type, use_row_layout> sorted_entries,
        tree_metadata metadata,
        key_type* tree
) {
    constexpr size_t node_size = 1u << node_size_log;

    const smallsize tid = blockDim.x * blockIdx.x + threadIdx.x;
    const smallsize slot = tid & (node_size - 1);
    smallsize nid = tid >> node_size_log;

    if (tid >= (metadata.total_nodes << node_size_log)) return;

    // determine which level we are on
    smallsize level = 0;
    for (; level < metadata.level_count; ++level) {
        smallsize nodes_on_level = metadata.nodes_on_level[level];
        if (nid < nodes_on_level) break;
        nid -= nodes_on_level;
    }
    smallsize level_stride = node_size;
    for (; level < metadata.level_count - 1; ++level) {
        level_stride *= node_size + 1;
    }
    const smallsize offset_within_level = (nid << node_size_log) + slot;

    // determine from where to load entry
    // for level L, level count LC, node size N, thread id tid
    // we compute offset i within level by subtracting previous level sizes from tid -> done in loop
    // we compute level stride S = N(N+1)^(LC-L-1) -> done in loop to circumvent ipow
    // at tree buffer position tid, load the key from position [S + i*S + floor(i/N)*S - 1]
    const smallsize load_offset = level_stride * (1 + offset_within_level + offset_within_level / node_size) - 1;

    // usually, we would write load_offset < size to check for valid bounds
    // but using the very last entry for an inner node would imply there is a valid child node to the right,
    // which can never be the case. so we explicitly exclude the last entry
    bool exists = load_offset < sorted_entries.size() - 1;

    // each thread writes exactly one entry
    key_type write = exists ? sorted_entries.extract_key(load_offset) : std::numeric_limits<key_type>::max();
    tree[tid] = write;
}


template <typename key_type, uint16_t node_size_log, bool caching_enabled, typename tile_type>
DEVICEQUALIFIER INLINEQUALIFIER
smallsize static_tree_search(
        tile_type tile,
        key_type lookup,
        const key_type* tree,
        tree_metadata metadata,
        const key_type* shmem_tree,
        smallsize cached_entries_count
) {
    constexpr size_t node_size = 1u << node_size_log;

    smallsize global_entries_offset = 0;
    smallsize local_entries_offset = 0;
    for (smallsize level = 0; level < metadata.level_count; ++level) {
        smallsize offset = global_entries_offset + local_entries_offset + tile.thread_rank();
        key_type entry;
        if (caching_enabled && offset < cached_entries_count) {
            entry = shmem_tree[offset];
        } else {
            entry = tree[offset];
        }
        const bool valid = entry != std::numeric_limits<key_type>::max();
        // find number of tree entries that are smaller than our searched key
        const bool gt = valid && lookup > entry;
        smallsize gt_mask = tile.ballot(gt);
        smallsize child_offset = __popc(gt_mask);
        local_entries_offset = local_entries_offset * (node_size + 1) + (child_offset << node_size_log);
        global_entries_offset += metadata.nodes_on_level[level] << node_size_log;
    }
    return local_entries_offset;
}


template <
        typename key_type,
        uint8_t opts,
        uint16_t node_size_log,
        uint16_t threads_per_block,
        uint16_t registers_per_thread,
        bool use_row_layout,
        bool range_query>
GLOBALQUALIFIER
void static_tree_lookup_kernel(
        mem::store_type<key_type, use_row_layout> sorted_entries,
        smallsize cached_entries_count,
        smallsize bytes_reserved_for_shuffle,
        const key_type* tree,
        tree_metadata metadata,
        const key_type* lookups,
        const key_type* upper_limits,
        smallsize* results,
        smallsize size
) {
    static_assert(node_size_log > 0, "node size cannot be 1");
    static_assert(node_size_log <= 5, "node size must not exceed 32");
    static_assert(threads_per_block <= 1024, "cannot have more than 1024 threads per block");

    constexpr size_t node_size = 1u << node_size_log;

    extern __shared__ uint8_t shared_memory[];

    using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, range_query>;
    using upper_type = typename algs::upper;

    const auto shmem_tree = reinterpret_cast<key_type*>(shared_memory + bytes_reserved_for_shuffle);

    const smallsize bid = blockIdx.x;
    const smallsize local_tid = threadIdx.x;
    const smallsize block_size = threads_per_block;
    const smallsize grid_size = threads_per_block * gridDim.x;

    if (opt::f::cache_upper_levels(opts)) {
        for (smallsize i = local_tid; i < cached_entries_count; i += block_size) {
            shmem_tree[i] = tree[i];
        }
        __syncthreads();
    }

    auto tile = cooperative_groups::tiled_partition<node_size>(cooperative_groups::this_thread_block());

    key_type local_lookups[registers_per_thread];
    upper_type local_upper_limits[registers_per_thread];
    uint32_t local_ranks[registers_per_thread];
    smallsize local_results[registers_per_thread];

    const smallsize elements_per_block = block_size * registers_per_thread;
    const smallsize grid_stride = grid_size * registers_per_thread;

    for (smallsize lookup_offset = bid * elements_per_block; lookup_offset < size; lookup_offset += grid_stride) {
        const auto this_block_count = std::min(elements_per_block, size - lookup_offset);

        #pragma unroll
        for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
            const auto block_rank = reg * block_size + local_tid;
            local_lookups[reg] = block_rank < this_block_count ? lookups[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            if constexpr (range_query) {
                local_upper_limits[reg] = block_rank < this_block_count ? upper_limits[lookup_offset + block_rank] : std::numeric_limits<key_type>::max();
            }
            local_ranks[reg] = block_rank;
        }

        if constexpr (opt::f::sort_lookups_block(opts)) {
            algs::sort(shared_memory).to_striped(local_lookups, local_ranks, local_upper_limits);
            __syncthreads();
        }

        #pragma unroll
        for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
            const auto block_rank = reg * block_size + local_tid;
            key_type key    = 0;
            key_type upper  = 0;
            smallsize value = not_found;
            bool to_find = false;
            if (block_rank < this_block_count) {
                key = local_lookups[reg];
                if constexpr (range_query) {
                    upper = local_upper_limits[reg];
                }
                to_find = true;
            }
            auto work_queue = tile.ballot(to_find);
            if (!work_queue) break;

            while (work_queue) {
                auto cur_rank = __ffs(work_queue) - 1;
                key_type cur_key = tile.shfl(key, cur_rank);
                key_type cur_upper;
                if constexpr (range_query) {
                    cur_upper = tile.shfl(upper, cur_rank);
                }
                smallsize cur_result;
                // actual lookup starts here
                {
                    auto offset = static_tree_search<key_type, node_size_log, opt::f::cache_upper_levels(opts)>(
                            tile, cur_key, tree, metadata, shmem_tree, cached_entries_count);
                    if constexpr (range_query) {
                        cur_result = mem::cooperative_range_query(tile, cur_key, cur_upper, offset, sorted_entries);
                    } else {
                        smallsize i = offset + tile.thread_rank();
                        bool found = i < sorted_entries.size() && sorted_entries.extract_key(i) == cur_key;
                        smallsize found_mask = tile.ballot(found);
                        if (!found_mask) {
                            cur_result = not_found;
                        } else {
                            smallsize hit = __ffs(found_mask) - 1;
                            cur_result = sorted_entries.extract_offset(offset + hit);
                        }
                    }
                    if (cur_rank == tile.thread_rank()) {
                        value   = cur_result;
                        to_find = false;
                    }
                }
                work_queue = tile.ballot(to_find);
            }

            if (block_rank < this_block_count) {
                if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
                    // delayed write
                    local_results[reg] = value;
                } else {
                    // immediate write
                    results[lookup_offset + local_ranks[reg]] = value;
                }
            }
        }

        if constexpr (opt::f::sort_unsort_lookups_block(opts)) {
            algs::result_shuffle(shared_memory).to_striped(local_results, local_ranks);
            __syncthreads();

            #pragma unroll
            for (smallsize reg = 0; reg < registers_per_thread; ++reg) {
                const auto block_rank = reg * block_size + local_tid;
                if (block_rank >= this_block_count) continue;
                results[lookup_offset + block_rank] = local_results[reg];
            }
        }
    }
}


template <
        typename key_type_,
        uint8_t opts,
        bool static_scheduling,
        bool use_row_layout,
        uint16_t node_size_log,
        uint16_t threads_per_block,
        uint16_t registers_per_thread>
class opt_static_tree {
public:
    using key_type = key_type_;

private:
    static constexpr size_t node_size = 1u << node_size_log;

    // override kernel setup for range queries
    // these parameters might not be ideal, this is just an educated guess
    static constexpr uint16_t rq_threads_per_block = 512;
    static constexpr uint16_t rq_registers_per_thread = 1;

    cuda_buffer<key_type> sorted_keys_buffer;
    cuda_buffer<smallsize> sorted_offsets_buffer;
    cuda_buffer<uint8_t> sorted_key_offset_pairs_buffer;
    size_t base_entries_count = 0;

    // buffer stores metadata and all tree levels in sequence
    cuda_buffer<key_type> tree_buffer;
    size_t shared_entries_count = 0;

    size_t shared_bytes_to_load = 0;
    size_t shared_bytes_for_shuffle = 0;

    uint32_t number_of_sms = 0;
    size_t max_shared_memory_bytes = 0;

    tree_metadata metadata;

    size_t shmem() {
        return shared_bytes_to_load + shared_bytes_for_shuffle;
    }

    mem::store_type<key_type, use_row_layout> make_store() {
        if constexpr (use_row_layout) {
            return mem::row_store<key_type>(sorted_key_offset_pairs_buffer.ptr(), base_entries_count);
        } else {
            return mem::col_store<key_type>(sorted_keys_buffer.ptr(), sorted_offsets_buffer.ptr(), base_entries_count);
        }
    }

public:
    static constexpr operation_support can_lookup = operation_support::async;
    static constexpr operation_support can_multi_lookup = operation_support::async;
    static constexpr operation_support can_range_lookup = operation_support::async;
    static constexpr operation_support can_update = operation_support::none;

    static std::string short_description() {
        std::string desc = "opt_static_tree";
        return desc;
    }

    static parameters_type parameters() {
        std::string opts_summary;
        opts_summary += opt::f::cache_upper_levels_partial(opts) ? "cp" : opt::f::cache_upper_levels(opts) ? "c" : "";
        opts_summary += opt::f::sort_unsort_lookups_block(opts) ? "su" : opt::f::sort_lookups_block(opts) ? "s" : "";
        opts_summary += opts_summary.empty() ? "none" : "";
        return {
                {"opts", opts_summary},
                {"schedule", static_scheduling ? "static" : "dynamic"},
                {"layout", use_row_layout ? "row" : "column"},
                {"node_size", std::to_string(node_size)},
                {"threads_per_block", std::to_string(threads_per_block)},
                {"registers_per_thread", std::to_string(registers_per_thread)},
                {"rq_threads_per_block", std::to_string(rq_threads_per_block)},
                {"rq_registers_per_thread", std::to_string(rq_registers_per_thread)}
        };
    }

    static size_t estimate_build_bytes(size_t size) {
        size_t base_bytes = (sizeof(smallsize) + sizeof(key_type)) * size;
        size_t sort_aux_bytes = sizeof(smallsize) * size + find_pair_sort_buffer_size<key_type, smallsize>(size);
        size_t tree_bytes = base_bytes >> (node_size_log - 1);
        size_t layout_transform_bytes = use_row_layout ? base_bytes : 0;
        return std::max(base_bytes + sort_aux_bytes, base_bytes + tree_bytes + layout_transform_bytes);
    }

    size_t gpu_resident_bytes() {
        // note that some of these buffers are never allocated
        return sorted_keys_buffer.size_in_bytes() +
            sorted_offsets_buffer.size_in_bytes() +
            sorted_key_offset_pairs_buffer.size_in_bytes() +
            tree_buffer.size_in_bytes();
    }

    void build(const key_type* keys, size_t size, size_t max_size, size_t available_memory_bytes, double* build_time_ms, size_t* build_bytes) {
        if (estimate_build_bytes(size) > available_memory_bytes) throw std::runtime_error("insufficient memory");

        constexpr size_t key_size = sizeof(key_type);
        constexpr size_t row_stride = sizeof(key_type) + sizeof(smallsize);

        number_of_sms = get_number_of_sms();
        max_shared_memory_bytes = get_max_optin_shmem_per_block_bytes();
        base_entries_count = size;

        // sort keys
        size_t temp_storage_bytes = 0;
        {
            cuda_buffer<uint8_t> temp_buffer;
            cuda_buffer<smallsize> offsets_buffer;

            offsets_buffer.alloc(size); C2EX
            init_offsets(offsets_buffer.ptr(), size, build_time_ms);

            sorted_keys_buffer.alloc(size); C2EX
            sorted_offsets_buffer.alloc(size); C2EX

            cudaDeviceSynchronize(); C2EX

            temp_storage_bytes = find_pair_sort_buffer_size<key_type, smallsize>(size);
            temp_buffer.alloc(temp_storage_bytes); C2EX
            timed_pair_sort(
                temp_buffer.raw_ptr, temp_storage_bytes,
                keys, sorted_keys_buffer.ptr(), offsets_buffer.ptr(), sorted_offsets_buffer.ptr(), size, build_time_ms);

            cudaDeviceSynchronize(); C2EX
        }

        // figure out tree structure
        metadata.level_count = 0;
        metadata.total_nodes = 0;
        {
            size_t level_count = 0;
            size_t nodes_on_current_level = SDIV(size, node_size);
            while (nodes_on_current_level > 1) {
                // one inner node covers (node_size + 1) nodes on lower level
                nodes_on_current_level = SDIV(nodes_on_current_level, node_size + 1);
                level_count++;
            }
            if (level_count > tree_metadata::max_level_count) {
                throw std::runtime_error("maximum tree height exceeded, choose a larger node size");
            }
            metadata.level_count = level_count;
            nodes_on_current_level = SDIV(size, node_size);
            for (size_t i = 0; i < level_count; i++) {
                nodes_on_current_level = SDIV(nodes_on_current_level, node_size + 1);
                metadata.total_nodes += nodes_on_current_level;
                metadata.nodes_on_level[metadata.level_count - 1 - i] = nodes_on_current_level;
            }
        }

        // reserve shmem for sort/shuffle
        if constexpr (opt::f::sort_lookups_block(opts)) {
            using algs = opt::algorithms<key_type, threads_per_block, registers_per_thread, false>;
            using rq_algs = opt::algorithms<key_type, rq_threads_per_block, rq_registers_per_thread, true>;

            shared_bytes_for_shuffle = std::max(algs::temp_storage_bytes, rq_algs::temp_storage_bytes);
            // add padding so that shared_bytes_for_shuffle is a multiple of 8 bytes
            shared_bytes_for_shuffle = (shared_bytes_for_shuffle + 7) & ~size_t(7);
        }

        tree_buffer.alloc(metadata.total_nodes * node_size); C2EX

        // only extend shared memory if cache_upper_levels is enabled
        if constexpr (opt::f::cache_upper_levels(opts)) {
            size_t remaining_bytes = max_shared_memory_bytes - shared_bytes_for_shuffle;
            size_t max_node_count = remaining_bytes / (node_size * key_size);

            size_t shared_node_count = 0;
            if (metadata.total_nodes <= max_node_count) {
                // if we can, cache everything
                shared_node_count = metadata.total_nodes;
            } else if (opt::f::cache_upper_levels_partial(opts)) {
                // otherwise, cache as much as we can fit
                shared_node_count = max_node_count;
            } else {
                // or only add full levels until we run out of space
                for (size_t i = 0; i < metadata.level_count; ++i) {
                    if (metadata.nodes_on_level[i] > max_node_count) break;
                    shared_node_count += metadata.nodes_on_level[i];
                    max_node_count -= metadata.nodes_on_level[i];
                }
            }
            shared_entries_count = node_size * shared_node_count;
            shared_bytes_to_load = shared_entries_count * key_size;

            // point query
            cudaFuncSetAttribute(
                    static_tree_lookup_kernel<key_type, opts, node_size_log, threads_per_block, registers_per_thread, use_row_layout, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem()); C2EX
            // range query
            cudaFuncSetAttribute(
                    static_tree_lookup_kernel<key_type, opts, node_size_log, rq_threads_per_block, rq_registers_per_thread, use_row_layout, true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem()); C2EX
        }

        // build tree
        {
            scoped_cuda_timer timer(0, build_time_ms);
            auto nb = SDIV(metadata.total_nodes * node_size, MAXBLOCKSIZE);
            build_static_tree_kernel<key_type, node_size_log, false><<<nb, MAXBLOCKSIZE>>>
                (mem::col_store(sorted_keys_buffer.ptr(), nullptr, size), metadata, tree_buffer.ptr());
        }

        // transform to row layout if necessary
        printf("opt_static_tree: built tree with %u levels, %u nodes, %zu size %u \n",
               metadata.level_count, metadata.total_nodes, metadata.total_nodes * node_size, size);

        if constexpr (use_row_layout) {
        sorted_key_offset_pairs_buffer.alloc(row_stride * size); C2EX
            transform_into_row_layout(
                sorted_keys_buffer.ptr(),
                sorted_offsets_buffer.ptr(),
                sorted_key_offset_pairs_buffer.ptr(),
                size,
                build_time_ms);
            sorted_keys_buffer.free();
            sorted_offsets_buffer.free();
        }

        size_t layout_transform_bytes = use_row_layout ? row_stride * size : 0;
        if (build_bytes) *build_bytes += row_stride * size + std::max<size_t>(
                temp_storage_bytes + sizeof(smallsize) * size,
                tree_buffer.size_in_bytes() + layout_transform_bytes);

        cudaDeviceSynchronize(); C2EX
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_static_tree_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, threads_per_block * registers_per_thread);
        static_tree_lookup_kernel
                <key_type, opts, node_size_log, threads_per_block, registers_per_thread, use_row_layout, false>
                <<<number_of_blocks, threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_entries_count, shared_bytes_for_shuffle,
            tree_buffer.ptr(), metadata,
            keys, nullptr, result, size);
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {

        nvtx3::scoped_range_in<nvtx_opt_static_tree_domain> launch{"launch"};

        size_t number_of_blocks = static_scheduling ? number_of_sms : SDIV(size, rq_threads_per_block * rq_registers_per_thread);
        static_tree_lookup_kernel
                <key_type, opts, node_size_log, rq_threads_per_block, rq_registers_per_thread, use_row_layout, true>
                <<<number_of_blocks, rq_threads_per_block, shmem(), stream>>>(
            make_store(),
            shared_entries_count, shared_bytes_for_shuffle,
            tree_buffer.ptr(), metadata,
            lower, upper, result, size);
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {
        range_lookup_sum(keys, keys, result, size, stream);
    }

//---------------
//OUTput to CSV FIle for DEbugging -rosina

    bool dump_leaves_csv(const std::string& path) const {
        if (base_entries_count == 0) return false;

        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        if (!ofs) return false;

        ofs << "leaf_id,local_idx,global_idx,key,offset\n";

        const size_t leaf_cnt = (base_entries_count + node_size - 1) / node_size;

        // Host views
        const size_t row_stride = sizeof(key_type) + sizeof(smallsize);
        const bool use_row = use_row_layout;

        std::vector<uint8_t> h_pairs;      // AoS
        std::vector<key_type> h_keys;      // SoA
        std::vector<smallsize> h_offs;     // SoA

        if (use_row) {
            h_pairs.resize(row_stride * base_entries_count);
            if (sorted_key_offset_pairs_buffer.ptr() == nullptr) return false;
            if (cudaMemcpy(h_pairs.data(), sorted_key_offset_pairs_buffer.ptr(),
                           h_pairs.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        } else {
            h_keys.resize(base_entries_count);
            h_offs.resize(base_entries_count);
            if (sorted_keys_buffer.ptr() == nullptr || sorted_offsets_buffer.ptr() == nullptr) return false;
            if (cudaMemcpy(h_keys.data(), sorted_keys_buffer.ptr(),
                           h_keys.size() * sizeof(key_type), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            if (cudaMemcpy(h_offs.data(), sorted_offsets_buffer.ptr(),
                           h_offs.size() * sizeof(smallsize), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        }

        // Stream rows
        for (size_t leaf = 0; leaf < leaf_cnt; ++leaf) {
            const size_t g_begin = leaf * node_size;
            const size_t g_end   = std::min(g_begin + node_size, base_entries_count);
            for (size_t gi = g_begin, local = 0; gi < g_end; ++gi, ++local) {
                key_type k{};
                smallsize o{};
                if (use_row) {
                    const uint8_t* p = h_pairs.data() + gi * row_stride;
                    std::memcpy(&k, p, sizeof(key_type));
                    std::memcpy(&o, p + sizeof(key_type), sizeof(smallsize));
                } else {
                    k = h_keys[gi];
                    o = h_offs[gi];
                }
                ofs << leaf << ',' << local << ',' << gi << ',' << k << ',' << o << '\n';
            }
        }
        return true;
    }

    bool dump_inner_csv(const std::string& path) const {
        if (metadata.level_count == 0 || metadata.total_nodes == 0) return false;
        if (tree_buffer.ptr() == nullptr) return false;

        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        if (!ofs) return false;

        ofs << "level,node_id,key_idx,key\n";

        const size_t total_entries = static_cast<size_t>(metadata.total_nodes) * node_size;
        std::vector<key_type> h_inner(total_entries);

        if (cudaMemcpy(h_inner.data(), tree_buffer.ptr(),
                       total_entries * sizeof(key_type), cudaMemcpyDeviceToHost) != cudaSuccess) return false;

        size_t cursor = 0;
        for (size_t lvl = 0; lvl < metadata.level_count; ++lvl) {
            const size_t nodes_on_lvl = metadata.nodes_on_level[lvl];
            for (size_t n = 0; n < nodes_on_lvl; ++n) {
                const size_t base = cursor + n * node_size;
                for (size_t i = 0; i < node_size; ++i) {
                    ofs << lvl << ',' << n << ',' << i << ',' << h_inner[base + i] << '\n';
                }
            }
            cursor += nodes_on_lvl * node_size;
        }
        return true;
    }

    // Convenience wrapper: writes <prefix>_leaves.csv and <prefix>_inner.csv
    bool dump_tree_csv(const std::string& prefix) const {
        const bool a = dump_leaves_csv(prefix + "_leaves.csv");
        const bool b = dump_inner_csv(prefix + "_inner.csv");
        return a || b;
    }



//----------------


    void destroy() {
        sorted_keys_buffer.free();
        sorted_offsets_buffer.free();
        sorted_key_offset_pairs_buffer.free();
        tree_buffer.free();
    }
};


template <typename key_type, bool use_row_layout = true>
using opt_static_tree_4090 = opt_static_tree<key_type, opt::f::none, false, use_row_layout, 4, 256, 4>;

#endif
