// merge: Search within a node using a cooperative thread tile; if found, delete in-place via left-shift.
// Reads each thread's key/offset at most once, then reuses registers for shifting.
// Returns true if the key was found and deleted; false otherwise.

#ifndef DEVICEQUALIFIER
#define DEVICEQUALIFIER __device__ __forceinline__
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

// Assumes helpers exist:
// - extract_key_node<key_type>(buffer, index_1_based)
// - extract_offset_node<key_type>(buffer, index_1_based)
// - set_key_node<key_type>(buffer, index_1_based, key)
// - set_offset_node<key_type>(buffer, index_1_based, off)
// - cg::extract<T>(buffer, byte_offset)
// - cg::set<T>(buffer, byte_offset, value)
// - tombstone sentinel

template <typename key_type>
DEVICEQUALIFIER bool merge(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
    key_type search_key)
{
    const smallsize tid = tile.thread_rank();

    // Read current logical size once from node header (by lane 0), then broadcast.
    smallsize num_elements = 0;
    if (tid == 0)
        num_elements = cg::extract<smallsize>(curr_node, sizeof(key_type));
    num_elements = tile.shfl(num_elements, 0);

    // Fast exit if empty.
    if (num_elements == 0)
        return false;

    // Each thread reads its key/offset once (if within bounds), stores in registers.
    key_type my_key = 0;
    smallsize my_off = 0;
    bool in_bounds = (tid < num_elements);

    if (in_bounds)
    {
        my_key = extract_key_node<key_type>(curr_node, tid + 1);
        if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
        {
            my_off = extract_offset_node<key_type>(curr_node, tid + 1);
        }
    }

    // Parallel membership test with ballot; compute first lane that matches.
    const bool is_match = in_bounds && (my_key == search_key);

    if (!tile.any(is_match))
        return false; // Not found; no further work.

    // Find first active match lane (lowest index)
    const unsigned mask = tile.ballot(is_match);
    const int first_active = __ffs(mask) - 1; // 1-based -> 0-based

    // Compute 1-based deletion index and broadcast.
    smallsize del_index = 0; // 1-based
    if (tid == first_active)
        del_index = static_cast<smallsize>(tid + 1);
    del_index = tile.shfl(del_index, first_active);

    // Deleting last position: just zero and decrement size.
    if (del_index == num_elements)
    {
        if (tid == 0)
        {
            set_key_node<key_type>(curr_node, del_index, static_cast<key_type>(0));
            set_offset_node<key_type>(curr_node, del_index, 0);
            smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
            cg::set<smallsize>(curr_node, sizeof(key_type), static_cast<smallsize>(curr_size - 1));
        }
        tile.sync();
        return true;
    }

    // For in-place left shift, lanes tid in [del_index, num_elements-1] write to position tid (1-based -> tid),
    // using the value originally read from position tid+1 (already in registers as my_key/my_off).
    const bool do_shift = (tid >= (del_index) && tid < num_elements);

    tile.sync(); // Ensure all reads completed before writes.

    if (do_shift)
    {
        set_key_node<key_type>(curr_node, tid, my_key);
        set_offset_node<key_type>(curr_node, tid, my_off);
    }

    tile.sync(); // Ensure shift completed prior to tail cleanup.

    if (tid == 0)
    {
        // Zero out the logical last element and decrement size.
        set_key_node<key_type>(curr_node, num_elements, static_cast<key_type>(0));
        set_offset_node<key_type>(curr_node, num_elements, 0);
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), static_cast<smallsize>(curr_size - 1));
    }

    tile.sync();
    return true;
}
