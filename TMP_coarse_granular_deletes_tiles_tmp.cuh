#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_DELETES_TILES_TMP_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_DELETES_TILES_TMP_CUH

// #include <cooperative_groups.h>
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_inserts.cuh"
//--> circular #include "coarse_granular_inserts_tiles.cuh"
#include "definitions_coarse_granular.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"
// #define TILE_SIZE 8 //

// namespace coop_g = cooperative_groups;

// New: Single-threaded perform_shift_insert (no tile)
template <typename key_type>
DEVICEQUALIFIER void perform_delete_insert_single(
    void *curr_node,
    smallsize insert_index,
    smallsize num_elements,
    key_type insertkey,
    smallsize thisoffset)
{
    // Shift elements up by 1 position starting from the end
    for (smallsize i = num_elements; i >= insert_index; --i)
    {
        key_type key = extract_key_node<key_type>(curr_node, i);
        smallsize offset = extract_offset_node<key_type>(curr_node, i);

        set_key_node<key_type>(curr_node, i + 1, key);
        set_offset_node<key_type>(curr_node, i + 1, offset);
    }

    // Insert the new key and offset
    set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
    set_offset_node<key_type>(curr_node, insert_index, thisoffset);

    // Increment node size
    smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
    cg::set<smallsize>(curr_node, sizeof(key_type), curr_size + 1);
}

template <typename key_type>
DEVICEQUALIFIER void perform_shift_delete_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node,
    smallsize del_index,
    smallsize num_elements,
    key_type insertkey)
{
    smallsize tid = tile.thread_rank();
    key_type key = 0;
    smallsize offset = 0;

    // Early Exit if deleting at last position
    if (del_index == num_elements)
    {
        if (tid == 0)
        {
            // set_key_node<key_type>(curr_node, insert_index, static_cast<key_type>(insertkey));
            // set_offset_node<key_type>(curr_node, insert_index, thisoffset);

            set_key_node<key_type>(curr_node, del_index, static_cast<key_type>(0));
            set_offset_node<key_type>(curr_node, del_index, 0);

            smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size - 1);
        }
        tile.sync();
        return;
    }

    bool do_shift = (tid >= del_index && tid < num_elements);
    //---->> HERE tid is never 16. last key is never grabbed.
    //----> itd is from 0 to 15

    if (do_shift)
    {
        key = extract_key_node<key_type>(curr_node, tid + 1);
        offset = extract_offset_node<key_type>(curr_node, tid + 1);
    }

    tile.sync(); // <-- everyone syncs

    if (do_shift)
    {
        set_key_node<key_type>(curr_node, tid, key);
        set_offset_node<key_type>(curr_node, tid, offset);
    }

    tile.sync(); // <-- ensure all shifts done before del last key

    if (tid == 0)
    {
        // zero out the last position
        // smallsize tile_meta_rank = tile.meta_group_rank();

        set_key_node<key_type>(curr_node, num_elements, static_cast<key_type>(0));
        set_offset_node<key_type>(curr_node, num_elements, 0);

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        cg::set<smallsize>(curr_node, sizeof(key_type), curr_size - 1);

        /* if (tile_meta_rank == 13) {
           printf("AFT Zero Out  %llu, DEL Index : %u,   size of node %u \n",
           static_cast<unsigned long long>(key), del_index,  num_elements);
           print_node<key_type>(curr_node, 16);
       } */
    }

    tile.sync(); // <-- ensure all threads have completed their shifts before exiting
}


//////----------------------------------------

template <typename key_type>
DEVICEQUALIFIER bool merge(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void *curr_node, smallsize curr_size
    key_type search_key)
{
    const smallsize tid = tile.thread_rank();

    // Read current logical size once from node header (by lane 0), then broadcast.
    smallsize num_elements = curr_size;
    //if (tid == 0)
    //    num_elements = cg::extract<smallsize>(curr_node, sizeof(key_type));
    //num_elements = tile.shfl(num_elements, 0);

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

///---------------------------------------

template <typename key_type>
DEVICEQUALIFIER void process_deletes_tile_merge(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize this_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    int total_keys = maxindex - minindex + 1;
    void *curr_node = starting_node;

    for (smallsize i = minindex; i <= maxindex; ++i)
    {
        key_type key = update_list[i];

        if (tile.thread_rank() == 0)
        {
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);

                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during insert", key, tile_id, this_tid);
                }

                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize del_index = 1;

        bool found = merge<key_type>(tile, curr_node, curr_size, key);

        if (!found)
        {
            if (tile.thread_rank() == 0)
            {
                printf("DEBUG_DEL_TILE: NOT FOUND After search: key %llu, del Index : %u, Tile ID: %u, found %u,  size of node %u \n",
                       static_cast<unsigned long long>(key), del_index, tile_id, found, curr_size);
            }
            continue;
        }
    }

#ifdef PRINT_PROCESS_DELETES_END
    __syncthreads();
    if (this_tid == 0 && tile_id == 0 && blockIdx.x == 0)
    {
        printf("END TILE DELS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, this_tid);
    }
#endif
}

///-------------------------------------------

template <typename key_type>
DEVICEQUALIFIER void process_deletes_tile(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize this_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    int total_keys = maxindex - minindex + 1;
    void *curr_node = starting_node;

    for (smallsize i = minindex; i <= maxindex; ++i)
    {
        key_type key = update_list[i];

        if (tile.thread_rank() == 0)
        {
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);

                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during insert", key, tile_id, this_tid);
                }

                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize del_index = 1;

        bool found = search_only_cuda_buffer_with_tile<key_type>(tile, curr_node, key, curr_size, del_index);

        if (!found)
        {
            if (tile.thread_rank() == 0)
            {
                printf("DEBUG_DEL_TILE: NOT FOUND After search: key %llu, del Index : %u, Tile ID: %u, found %u,  size of node %u \n",
                       static_cast<unsigned long long>(key), del_index, tile_id, found, curr_size);
            }
            continue;
        }

        /* Begin: this can be removed */
        key_type key_at_index = extract_key_node<key_type>(curr_node, del_index);

        if (key_at_index != key)
        {
            if (tile.thread_rank() == 0)
            {
                printf("ERROR NOT THE KEY WE EXPECTED: %llu != %llu at index %u, Tile ID: %u, Thread ID: %u\n",
                       static_cast<unsigned long long>(key_at_index), static_cast<unsigned long long>(key), del_index, tile_id, this_tid);
            }
            return;
        }

        /* End: this can be removed */

        if (del_index > curr_size || del_index < 1)
        {
            printf("ERROR: Invalid del_index %u with curr_size %u\n", del_index, curr_size);
            return;
        }

        if (key_at_index == key)
        {
            perform_shift_delete_tile<key_type>(tile, curr_node, del_index, curr_size, key);
        }
    }

#ifdef PRINT_PROCESS_DELETES_END
    __syncthreads();
    if (this_tid == 0 && tile_id == 0 && blockIdx.x == 0)
    {
        printf("END TILE DELS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, this_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_deletes_tile_bulk(
    key_type maxkey,
    smallsize minindex,
    smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    const smallsize lastpos_off = get_lastposition_bytes<key_type>(node_size);
    const smallsize my_tid = tile.thread_rank();
    const smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();
    smallsize this_tid = my_tid;
    void *curr_node = starting_node;
    smallsize size_del_list = maxindex - minindex + 1;

    for (smallsize i = minindex; i <= maxindex; ++i)
    {
        const key_type key = update_list[i];

        // Tile leader walks the node chain until node's max >= key
        if (tile.thread_rank() == 0)
        {
            key_type curr_max = cg::extract<key_type>(curr_node, 0);
            while (curr_max < key)
            {
                const smallsize next_ptr = cg::extract<smallsize>(curr_node, lastpos_off);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_count)
                {
                    ERROR_INSERTS("ERROR: invalid link in node chain during delete", key, tile_id, this_tid);
                    break;
                }
                curr_node = static_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        // Broadcast current node pointer from lane 0 to the tile
        uintptr_t p = reinterpret_cast<uintptr_t>(curr_node);
        p = tile.shfl(p, 0);
        curr_node = reinterpret_cast<void *>(p);
        tile.sync();

        const smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        smallsize del_index = 1;
        //------------------------------------

        {
            // Snapshot per-lane node key once. Why: avoid reloading inside the scan loop.
            key_type my_key = 0;
            smallsize my_offset;
            bool lane_live = false; // why: ignore empty/tombstone slots

            if (my_tid < curr_size)
            {
                my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
                my_offset = extract_offset_node<smallsize>(curr_node, my_tid + 1);
                if (my_key != key_type(0) && my_key != static_cast<key_type>(tombstone))
                {
                    lane_live = true;
                }
            }
            tile.sync();

            // Tile-wide mask of lanes whose key matches ANY of the next K delete keys
            unsigned match_mask = 0u; // why: single mask accumulates ballots over the scan window

            smallsize scan_i = i;          // start at current outer-loop index
            smallsize checks = 0;          // how many delete keys examined
            const smallsize K = curr_size; // why: per your note, tile_size == node_size

            while (checks < K && scan_i <= maxindex)
            {
                const key_type del_key = update_list[scan_i];

                const bool eq = lane_live && (my_key == del_key);
                const unsigned b = tile.ballot(eq); // same on all lanes
                match_mask |= b;                    // using OR accumulate across delete keys

                ++scan_i;
                ++checks;
            }

            // Optionally compute helpers for the next phase (still no deletes):
            const unsigned any_mask = match_mask; // identical on all lanes
            const unsigned invert_mask = ~any_mask;
            const smallsize match_count = __popc(any_mask);
            const smallsize first_lane = any_mask ? (static_cast<smallsize>(__ffs(any_mask) - 1))
                                                  : node_size + 1;
            int next_one_lane = -1;

            /*
            { //compaction
                // Fast exit if no matches
                if (any_mask != 0u)
                {
                    // check special case invert mask is ==0 meaning all keys are deleted
                    if (lane_live && invert_mask != 0u) {

                       smallsize first_zero = static_cast<smallsize>(__ffs(invert_mask) - 1);
                       if (first_zero > first_lane)
                       {
                          //insert at the first_lane positino
                          if (this_tid == first_zero){
                            set_key_node<key_type>(curr_node, first_lane +1, my_key);
                            set_offset_node<smallsize>(curr_node, first_lane + 1, my_offset);
                          }
                        }

                        next_one_lane = static_cast<smallsize>(__ffs(any_mask >> (first_zero + 1)) - 1);
                        if(next_one_lane == -1)
                        {
                           //only all zero's remaining.
                           // shift left all keys remaining after first_zero up until curr_size
                           if( my_tid >= (first_zero + 1) &&  my_tid < curr_size)
                           {
                               set_key_node<key_type>(curr_node, first_lane + (my_tid-first_zero) + 1, my_key);
                               set_offset_node<smallsize>(curr_node, first_lane + (my_tid-first_zero) + 1, my_offset);
                           }
                           curr_size = curr_size - (first_zero -first_lane); // How many del keys were in between the two

                           continue;

                        } else{ // we have more del keys in the node

                         next_one_lane += first_zero + 1; // adjust lane index


                        }


                    }


                }
                //....

            }  // compaction
              */
        }

        /*       shift compaction from Justus

        bool is_set = my_key != EMPTY;
        uint32_t set_mask = ballot(is_set)
        uint32_t set_count = ___popc(is_set)

        uint32_t source = __fns(set_mask, 0, rank+1)

        key_type shuffled = shfl(my_key, source)
        my_key = rank < set_count ? shuffled : EMPTY

        */

        /*
        // Build search mask: lanes AFTER first_lane, within [0..curr_size-1], live, and NOT matched
        const unsigned live_mask = tile.ballot(lane_live);

        // bits >= (first_lane+1)
        const unsigned after_mask = (first_lane_local + 1u >= 32u)
                                        ? 0u
                                        : (0xFFFFFFFFu << (first_lane_local + 1u));
        // bits < curr_size
        const unsigned within_size = (curr_size >= 32)
                                         ? 0xFFFFFFFFu
                                         : ((1u << curr_size) - 1u);

        const unsigned zeros_after = (~any_mask) & live_mask & after_mask & within_size;
        const int next_zero_lane = __ffs(zeros_after) - 1; // -1 if none

        if (next_zero_lane >= 0 && static_cast<int>(this_tid) == next_zero_lane)
        {
            // Node slots are 1-based; write promoted element to position first_lane+1
            set_key_node<key_type>(curr_node, static_cast<smallsize>(first_lane_local + 1), my_key);
            set_offset_node<smallsize>(curr_node, static_cast<smallsize>(first_lane_local + 1), my_offset);
        }
        */
        //           }
        //         tile.sync();
        //     }

        //     i = scan_i - 1;
        //  }
    }

#ifdef PRINT_PROCESS_DELETES_END
    __syncthreads(); // only valid if all threads in block reach here
    if (this_tid == 0 && tile_id == 0 && blockIdx.x == 0)
    {
        printf("END TILE DELS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, this_tid);
    }
#endif
}

#endif

/*


            (if first_lane == node_size +1 meaning no more 1s in the mask, we exit)
            // ( -->step * :note of there are no remaining 1s in the mask OR if first_lane is greater than next_zero_lane, we still have work to do
            // we must move all keys at next_zero_lane +1 and onward upto curr_size... beside the previous first_lane position)

            // if we found another true position in the mask, then again, we now look for the first  zero position
            // again call this next_zero_lane and again do the above, copy this

// Not Tried yet
// Assumptions:
//  - Node layout matches your insert path:
//      * key(0)   : max key in node
//      * size     : at byte offset sizeof(key_type)
//      * keys 1..size are sorted ascending
//  - Helpers exist (same as in your insert path):
//      cg::extract<T>(ptr, offset), cg::set<T>(ptr, offset, value)
//      extract_key_node<T>(node, idx_1based)
//      extract_offset_node<T>(node, idx_1based)
//      set_key_node<T>(node, idx_1based, key)
//      set_offset_node<T>(node, idx_1based, off)
//      single_search_cuda_buffer_tile<T>(tile, node, key, size, out_index)
//  - TILE_SIZE matches coop_g::thread_block_tile<TILE_SIZE>.
//  - updatable_cg_params fields match your insert path.
//
// Why shift-left instead of tombstoning + compaction? It's simple and keeps
// the node tightly packed; you can layer a tombstone + prefix-sum compactor later.

#include <cooperative_groups.h>
namespace coop_g = cooperative_groups;

#ifndef DEVICEQUALIFIER
#define DEVICEQUALIFIER __device__ __forceinline__
#endif

// ---- helper: shift-left delete at index (1-based) ----
// Moves entries [delete_idx+1 .. curr_size] down by one. Updates size and max.
// Expects: 1 <= delete_idx <= curr_size and curr_size > 0.

template <typename key_type, int TILE_SIZE>
DEVICEQUALIFIER void perform_shift_delete_tile(
    coop_g::thread_block_tile<TILE_SIZE> tile,
    void* node,
    smallsize delete_idx,
    smallsize curr_size)
{
    const smallsize my_tid = static_cast<smallsize>(tile.thread_rank());
    const bool in_range = (my_tid + 1) <= curr_size; // valid lane for loads

    key_type    my_key    = 0;
    smallsize   my_offset = 0;

    if (in_range) {
        my_key    = extract_key_node<key_type>(node, my_tid + 1);
        my_offset = extract_offset_node<key_type>(node, my_tid + 1);
    }

    // Read before any writes
    tile.sync();

    // Move down if strictly after the deleted slot
    if (in_range && (my_tid + 1) > delete_idx) {
        set_key_node<key_type>(node, (my_tid + 1) - 1, my_key);
        set_offset_node<key_type>(node, (my_tid + 1) - 1, my_offset);
    }

    tile.sync();

    // Lane 0: shrink size, update max key in header, clear old tail (optional)
    if (my_tid == 0) {
        const smallsize new_size = static_cast<smallsize>(curr_size - 1);
        cg::set<smallsize>(node, sizeof(key_type), new_size);
        if (new_size > 0) {
            key_type new_max = extract_key_node<key_type>(node, new_size);
            cg::set<key_type>(node, 0, new_max);
            // Clear the old last slot (not strictly required once size is shrunk)
            set_key_node<key_type>(node, new_size + 1, key_type(0));
            set_offset_node<key_type>(node, new_size + 1, smallsize(0));
        } else {
            cg::set<key_type>(node, 0, key_type(0));
        }
    }

    tile.sync();
}

// ---- main bulk delete path ----

template <typename key_type, int TILE_SIZE>
DEVICEQUALIFIER void process_deletes_tile_bulk_hybrid(
    key_typemaxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params* launch_params,
    void* starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type*  update_list = static_cast<const key_type*>(launch_params->update_list);
    // Offsets not needed for delete; present for parity with inserts
    const smallsize* offset_list = static_cast<const smallsize*>(launch_params->offset_list);

    void*     allocation_buffer       = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride             = launch_params->node_stride;
    smallsize node_size               = launch_params->node_size;

    const smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    const smallsize my_tid             = static_cast<smallsize>(tile.thread_rank());

    // Identify our logical tile id (debug only)
    // smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize update_idx = minindex;
    void*     curr_node  = starting_node;

    // Read node header: max key in slot 0
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    while (update_idx <= maxindex) {
        key_type del_key = update_list[update_idx];

        // Lane 0 advances along the chain until this node can contain del_key
        if (my_tid == 0) {
            while (curr_max < del_key) {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count) {
                    // key not present in chain; nothing to delete in tail
                    break;
                }
                curr_node = reinterpret_cast<uint8_t*>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max  = cg::extract<key_type>(curr_node, 0);
            }
        }
        // Broadcast node pointer and max
        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr  = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void*>(raw_ptr);

        // Node size
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // If node empty, nothing to do here: try next key (it may be in later nodes)
        if (curr_size == 0) {
            if (my_tid == 0) { update_idx++; }
            update_idx = tile.shfl(update_idx, 0);
            continue;
        }

        // If key is larger than this node's max even after chain advance, skip key
        if (del_key > curr_max) {
            if (my_tid == 0) { update_idx++; }
            update_idx = tile.shfl(update_idx, 0);
            continue;
        }

        // Try to locate the key in this node cooperatively
        smallsize del_idx_1b = 0;
        bool found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, del_key, curr_size, del_idx_1b);

        // Any lane found?
        const bool key_found_any = tile.any(found);
        if (!key_found_any) {
            // Might be in an earlier node (if duplicates) or already deleted; move to next key
            if (my_tid == 0) { update_idx++; }
            update_idx = tile.shfl(update_idx, 0);
            continue;
        }

        // Broadcast delete index (from lane 0 after search cooperated)
        del_idx_1b = tile.shfl(del_idx_1b, 0);

        // Perform shift-left delete on this node
        perform_shift_delete_tile<key_type, TILE_SIZE>(tile, curr_node, del_idx_1b, curr_size);

        // Update local header mirrors
        curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));
        curr_max  = cg::extract<key_type>(curr_node, 0);

        // Move to next delete key
        if (my_tid == 0) { update_idx++; }
        update_idx = tile.shfl(update_idx, 0);

        // Optional: if after deletion, the next delete key is <= curr_max, we stay on this node
        // and continue the loop, otherwise chain-advance at the top will move us as needed.
    }
}

// ---- Optional: bulk delete multiple keys in the same node (range loop) ----
// For higher throughput, you can wrap the single-key delete in an inner while:
// while (update_idx <= maxindex && update_list[update_idx] <= curr_max) { ... }
// This is intentionally kept simple above for clarity and correctness.



*/