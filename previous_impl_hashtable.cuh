#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <optional>

#include "../warpcore/warpcore.cuh"
#include "definitions.cuh"


template<class wc_table>
GLOBALQUALIFIER
void warpcore_build_kernel(wc_table hash_table, const typename wc_table::key_type* keys, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    hash_table.insert(keys[gid], gid, group);
}


template<class wc_table>
GLOBALQUALIFIER
void warpcore_insert_kernel(wc_table hash_table, const typename wc_table::key_type* keys, const smallsize* offsets, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    hash_table.insert(keys[gid], offsets[gid], group);
}


template<class wc_table>
GLOBALQUALIFIER
void warpcore_delete_kernel(wc_table hash_table, const typename wc_table::key_type* keys, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    // note: this function is not part of warpcore proper, it was added later on
    hash_table.erase_all(keys[gid], group);
}


template<class wc_table>
GLOBALQUALIFIER
void warpcore_lookup_kernel(wc_table hash_table, const typename wc_table::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    typename wc_table::key_type key = keys[gid];
    smallsize retrieved_index;

    // note: this function is not part of warpcore proper, it was added later on
    const auto status = hash_table.retrieve(key, retrieved_index, group);
    if (group.thread_rank() == 0) {
        if (status.has_key_not_found() || status.has_probing_length_exceeded()) {
            result[gid] = not_found;
        } else {
            result[gid] = retrieved_index;
        }
    }
}


template<class wc_table>
GLOBALQUALIFIER
void warpcore_multi_lookup_kernel(wc_table hash_table, const typename wc_table::key_type* keys, smallsize* result, const smallsize num_in) {
    const size_t tid = global_thread_id();
    const smallsize gid = tid / wc_table::cg_size();
    const auto group = cooperative_groups::tiled_partition<wc_table::cg_size()>(cooperative_groups::this_thread_block());
    if (gid >= num_in) return;

    typename wc_table::key_type key = keys[gid];
    typename wc_table::index_type num_out; // ignored
    smallsize agg = 0;

    hash_table.for_each(
        [=, &agg] DEVICEQUALIFIER (const typename wc_table::key_type, const typename wc_table::value_type& retrieved_index, const typename wc_table::index_type) {
            agg += retrieved_index;
        },
        key,
        num_out,
        group
    );

    for (size_t i = wc_table::cg_size() >> 1u; i > 0; i >>= 1u) {
        agg += group.shfl_down(agg, i);
    }
    if (group.thread_rank() == 0) {
        result[gid] = agg;
    }
}


template <typename key_type_, uint8_t initial_load_percent = 80>
class hashtable {
public:
    using key_type = key_type_;

private:
    using wc_table = warpcore::MultiValueHashTable<key_type, smallsize, key_type(-2), key_type(-1)>;

    std::optional<wc_table> wrapped_table;

public:
    static constexpr bool can_lookup = true;
    static constexpr bool can_multi_lookup = true;
    static constexpr bool can_range_lookup = false;
    static constexpr bool can_update = true;

    static std::string short_description() {
        std::string desc = "warpcore";
        desc += "_" + std::to_string(initial_load_percent);
        return desc;
    }

    static size_t estimate_build_bytes(size_t size) {
        // hash table size will always be the next higher prime, use factor 1.2 as an approximation
        // https://projecteuclid.org/journals/proceedings-of-the-japan-academy-series-a-mathematical-sciences/volume-28/issue-4/On-the-interval-containing-at-least-one-prime-number/10.3792/pja/1195570997.full
        return static_cast<size_t>(1.2 * size * 100 / initial_load_percent) * (sizeof(key_type) + sizeof(smallsize));
    }

    size_t gpu_resident_bytes() {
        return wrapped_table.value().bytes_total();
    }

    void build(const key_type* keys, size_t size, double* build_time_ms, size_t* build_bytes) {
        const size_t capacity = static_cast<size_t>(size * 100.0 / initial_load_percent);
        wrapped_table.emplace(capacity);

        cuda_timer timer(0);
        timer.start();

        warpcore_build_kernel<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
            wrapped_table.value(),
            keys,
            size
        );

        timer.stop();
        if (build_time_ms) *build_time_ms += timer.time_ms();
        if (build_bytes) *build_bytes += gpu_resident_bytes();

        cudaDeviceSynchronize(); CUERR
    }

    void lookup(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        warpcore_lookup_kernel<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            keys,
            result,
            size
        );
    }

    void multi_lookup_sum(const key_type* keys, smallsize* result, size_t size, cudaStream_t stream) {

        warpcore_multi_lookup_kernel<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            keys,
            result,
            size
        );
    }

    void range_lookup_sum(const key_type* lower, const key_type* upper, smallsize* result, size_t size, cudaStream_t stream) {}

    void destroy() {
        wrapped_table.reset();
    }

    void insert(const key_type* update_list, const smallsize* offsets, size_t size, cudaStream_t stream) {

        warpcore_insert_kernel<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            update_list,
            offsets,
            size
        );
    }

    void remove(const key_type* update_list, size_t size, cudaStream_t stream) {

        warpcore_delete_kernel<<<SDIV(size * wc_table::cg_size(), MAXBLOCKSIZE), MAXBLOCKSIZE, 0, stream>>>(
            wrapped_table.value(),
            update_list,
            size
        );
    }
};


#endif
