// =============================================================================
// File: coarse_granular_inserts_tiles_bulk_opt2_logic.cuh
// Author: Rosina Kharal
// Description: Implements coarse_granular_inserts_tiles_bulk_opt2_logic
// Copyright (c) 2025 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================

#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_BULK_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_INSERTS_TILES_BULK_CUH

// #include <cooperative_groups.h>
#include "coarse_granular_deletes.cuh"
#include "coarse_granular_inserts.cuh"
#include "coarse_granular_deletes_tiles.cuh"
#include "definitions_updates.cuh"
#include "tile_utils.cuh"

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;

        smallsize free_space = node_size - curr_size;

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0)
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
            key_type next_key = 0;

            while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
            {
                next_key = update_list[update_idx];
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                if (next_key == testkey && !max_testkey)
                {
                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    update_idx++;
                    break;
                }
                else if (next_key > testkey)
                {
                    break;
                }
                else
                {
                    key = next_key;
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                    set_key_node<key_type>(curr_node, next_index, key);
                    set_offset_node<key_type>(curr_node, next_index, offset);

                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    insert_count++;
                    update_idx++;
                    next_index++;
                }
            }
            // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
        }

        tile.sync();
        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_hybrid(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;

        smallsize free_space = node_size - curr_size;
        //--------------------------------- REGULAR TILE SHIFT RIGHT LOGIC --------------------------------

        if (free_space < 2 || (total_keys - total_insert_count) < 2)
        {
            smallsize insert_idx = 0;
            bool search_found = single_search_cuda_buffer_tile<key_type>(tile, curr_node, key, curr_size, insert_idx);

            if (search_found)
            {
                update_idx++;
                continue;
            }

            key_type key_at_index = extract_key_node<key_type>(curr_node, insert_idx);

            // if (tile.thread_rank() == 0) DEBUG_PI_TILE("Not Found", key, key_at_index, insert_index);

            if (key_at_index < key)
            {
                // insert_index = (curr_size == 0) ? 1 : insert_index + 1;
                insert_idx = (curr_size == 0) ? 1 : insert_idx;
            }

            if (curr_size < node_size)
            {
                /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                    printf("DEBUG_PI_TILE: Before shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                }
                */
                perform_shift_insert_tile<key_type>(tile, curr_node, insert_idx, curr_size, key, offset);
                update_idx++;
                /* if (tile.thread_rank() == 0 && key==1235571559 ) {
                    printf("DEBUG_PI_TILE: After shift insert | Key: %llu, Insert Index: %u, Tile ID: %u, Thread ID: %u\n",static_cast<unsigned long long>(key), insert_index, tile_id, this_tid);
                    print_node<key_type>(curr_node, node_size);
                }*/
            }
            total_insert_count += 1;
            continue;
        }

        //--------------------------------- BULK INSERTION LOGIC --------------------------------

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0)
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
            key_type next_key = 0;

            while (update_idx <= maxindex && insert_count < free_space && next_index < node_size)
            {
                next_key = update_list[update_idx];
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                if (next_key == testkey && !max_testkey)
                {
                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    update_idx++;
                    break;
                }
                else if (next_key > testkey)
                {
                    break;
                }
                else
                {
                    key = next_key;
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                    set_key_node<key_type>(curr_node, next_index, key);
                    set_offset_node<key_type>(curr_node, next_index, offset);

                    // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                    insert_count++;
                    update_idx++;
                    next_index++;
                }
            }
            // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
        }

        tile.sync();
        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] < curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_tile_bulk_opt_2(
    key_type maxkey, smallsize minindex, smallsize maxindex,
    updatable_cg_params *launch_params,
    void *starting_node,
    coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;

    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();
   smallsize total_insert_count_across_all_nodes = 0;
    while (update_idx <= maxindex)
    {
       
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

       

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);


       // if (tile_id == 0 && my_tid == 0)  DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        //if (tile_id == 0 && my_tid == 0) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        if (tile_id == 0 && my_tid == 0) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0;
        key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;
        smallsize tid_old_test_key = 0;

        smallsize free_space = node_size - curr_size;


        //--------------------------------- BULK INSERTION LOGIC --------------------------------

       
        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }

        

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            if (my_tid == 0) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }
        tile.sync();
        bool insert_curr_node = true;
        key_found = false;
        smallsize added_keys = 0;


        //***************************************************** */
        // LOOP AROUND AND KEEP TRYING TO ADD INSERT KEYS TO CURR_NODE as LONG AS We CAN
        //***************************************************** */

       // if (tile_id == 0 && my_tid == 0) {
       //             DEBUG_PI_TILE_BULK("CurrNode Before Insert Loop", my_key, update_idx, maxindex);
       //             print_node<key_type>(curr_node, node_size);
        //
       // }

        while (insert_curr_node)
        {
            max_testkey = false;

            added_keys = 0;
            bool is_gt = active && (my_key > key);
            unsigned mask = tile.ballot(is_gt);

            if (mask != 0)
            {
                insert_index = __ffs(mask);
                testkey = tile.shfl(my_key, insert_index - 1);
                testoffset = tile.shfl(my_offset, insert_index - 1);
                tid_test_key = insert_index - 1;
            }
            else
            {

                insert_index = curr_size + 1;
                max_testkey = true;
                testkey = curr_max;
                tid_test_key = node_size + 1;
            }

            tile.sync();
           // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("InsertPos", key, testkey, is_gt);
           // if (tile_id == 0 && my_tid == 0)DEBUG_PI_TILE_BULK("TestKey", testkey, tid_test_key, insert_index);

            // before we insert the next key. update the current node with the original keys that need to come into place
            smallsize added_keys = 0;
            if (tid_old_test_key != tid_test_key && tid_old_test_key >= 0 && insert_count > 0)
            {
               // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals", my_tid, my_key, my_offset);
               // if (tile_id == 0) DEBUG_PI_TILE_BULK("Add originals2", insert_index, tid_old_test_key, tid_test_key);

                smallsize max_thread_id=(tid_test_key <= node_size)? tid_test_key : node_size;
                if (active && my_tid >= tid_old_test_key && my_tid < max_thread_id)

                {
                    set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
                    set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
                    added_keys = tid_test_key - tid_old_test_key;
                }
                //*******DEBUGGING*/
                /*
                if (tile_id == 0 && my_tid == tid_old_test_key) {
                    DEBUG_PI_TILE_BULK("Done Adding Originals", my_key, tid_old_test_key, tid_test_key);
                    print_node<key_type>(curr_node, node_size);

                } */
                //*******DEBUGGING*/

            }

            insert_index = insert_index + insert_count;
            smallsize next_index = insert_index + 1; // next place where we can begin inserting
            //insert_count = 0;
            if (my_tid == 0)
            {
                set_key_node<key_type>(curr_node, insert_index, key);
                set_offset_node<key_type>(curr_node, insert_index, offset);
                insert_count++;
                update_idx++;
                //*******DEBUGGING*/
                /* if (tile_id == 0 && my_tid == 0) {
                    DEBUG_PI_TILE_BULK("Done inserting one", key, insert_index, insert_index);
                    print_node<key_type>(curr_node, node_size); 

                } */
                //*******DEBUGGING*/

                key_type next_key = 0;

                // try to see if any more keys can fit in before TEST KEY
                while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
                {
                    next_key = update_list[update_idx];
                    offset = offset_list[update_idx];
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
                    // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

                    if (next_key == testkey && !max_testkey)
                    {
                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        update_idx++;
                        break;
                    }
                    else if (next_key > testkey)
                    {
                        break;
                    }
                    else
                    {
                        key = next_key;
                        // offset = offset_list[update_idx];
                        //  if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                        set_key_node<key_type>(curr_node, next_index, key);
                        set_offset_node<key_type>(curr_node, next_index, offset);

                        // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                        insert_count++;
                        update_idx++;
                        next_index++;
                    }
                }
                // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);
            }
            // INSERTED ONE FOR SURE and POSSIBLY MORE THAT WERE LESS THAN TEST KEY

            tile.sync();
            insert_count = tile.shfl(insert_count, 0);
            update_idx = tile.shfl(update_idx, 0);
            next_index = tile.shfl(next_index, 0);
            
        

            if (update_idx > maxindex || next_index > node_size || insert_count >= free_space)
            {
                insert_curr_node = false; // no more keys to insert in this node
                break;
            }
            key = update_list[update_idx];
            offset = offset_list[update_idx];

            if (key > curr_max)
            {
                insert_curr_node = false;
                break;
            } // no more keys in this node

            found = (active && my_key == key);
            key_found = tile.any(found);

            // exit if key found and re do loop
            while (key_found && key <= curr_max && update_idx <= maxindex)
            {
                //if (tile_id == 0) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
                // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
                update_idx++;
                key = update_list[update_idx];
                offset = offset_list[update_idx];
                found = (active && my_key == key);
                key_found = tile.any(found);
            }

            if (key_found || key > curr_max || update_idx > maxindex)
                insert_curr_node = false;

            // --- go to top
            tid_old_test_key = tid_test_key;

            tile.sync();
        }

        // Update TOTAL SIZE
        total_insert_count_across_all_nodes += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        //if (active && my_tid >= (insert_index - 1))
        if (active && my_tid >= (tid_test_key))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);
            //  DEBUG_PI_TILE_BULK("Bottom After While Before ADDING oRIGNS",my_tid, my_key, update_idx, insert_index);
            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);

              /* if (tile_id == 0 && my_tid == insert_index - 1) {
                    DEBUG_PI_TILE_BULK("Bottom After While ADDED oRIGNS", my_key, update_idx, maxindex);
                    print_node<key_type>(curr_node, node_size);

                } */
        }
      
        tile.sync();

       
        if (curr_size == node_size && update_idx <= maxindex &&
            (total_keys - total_insert_count_across_all_nodes > 0) &&
            (update_list[update_idx] <= curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }
    }

    // tile.sync();
    // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

#endif

/* 

    
    template <typename key_type>
    DEVICEQUALIFIER void process_inserts_tile_bulk_opt(
        key_type maxkey, smallsize minindex, smallsize maxindex,
        updatable_cg_params *launch_params,
        void *starting_node,
        coop_g::thread_block_tile<TILE_SIZE> tile)
{
    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize node_stride = launch_params->node_stride;
    smallsize node_size = launch_params->node_size;

    smallsize partition_count = launch_params->partition_count_with_overflow;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize my_tid = tile.thread_rank();
    smallsize tile_id = blockIdx.x * (blockDim.x / tile.size()) + tile.meta_group_rank();

    smallsize total_keys = maxindex - minindex + 1;
    smallsize total_insert_count = 0;
    void *curr_node = starting_node;
    smallsize update_idx = minindex;
    key_type curr_max = cg::extract<key_type>(curr_node, 0);

    // if (tile_id == 1) DEBUG_PI_TILE_BULK("Before Loop", curr_max, minindex, maxindex);
    // tile.sync();

    while (update_idx <= maxindex)
    {
        key_type key = update_list[update_idx];
        smallsize offset = offset_list[update_idx];

        // if (key ==1248854015) DEBUG_PI_TILE_BULK("Insert Key", key, offset, update_idx);
        // if (key ==1248854015 ) DEBUG_PI_TILE_BULK("Thread Info", my_tid, tile_id, curr_max);

        if (tile.thread_rank() == 0)
        {
            while (curr_max < key)
            {
                smallsize next_ptr = cg::extract<smallsize>(curr_node, lastposition_bytes);
                if (next_ptr == 0 || (next_ptr - 1) >= allocation_buffer_count)
                {
                    ERROR_INSERTS("Invalid link in chain", key, tile_id, my_tid);
                }
                /// if (tile_id == 1) DEBUG_PI_TILE_BULK("Next Node", key, next_ptr, my_tid);
                curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + (next_ptr - 1) * node_stride;
                curr_max = cg::extract<key_type>(curr_node, 0);
            }
        }

        curr_max = tile.shfl(curr_max, 0);
        uintptr_t raw_ptr = reinterpret_cast<uintptr_t>(curr_node);
        raw_ptr = tile.shfl(raw_ptr, 0);
        curr_node = reinterpret_cast<void *>(raw_ptr);

        // tile.sync();

        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // need to check if space in the node for new keys
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Current Node Size", curr_size, node_size, my_tid);
        if (curr_size >= node_size)
        {
            // DEBUG_PI_TILE_BULK("Node size exceeded", key, curr_size, node_size);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);

            continue;
        }

        smallsize insert_index = 1;
        smallsize insert_count = 0;
        key_type testkey = 0;
        smallsize testoffset = 0 key_type my_key = 0;
        smallsize my_offset = 0;
        bool active = false;
        bool found = false;
        bool max_testkey = false;
        smallsize tid_test_key = node_size + 1;

        smallsize free_space = node_size - curr_size;

        if (my_tid < curr_size)
        {
            my_key = extract_key_node<key_type>(curr_node, my_tid + 1);
            my_offset = extract_offset_node<key_type>(curr_node, my_tid + 1);
            if (my_key != 0 && my_key != static_cast<key_type>(tombstone))
            {
                active = true;
                found = (my_key == key);
            }
        }
        tile.sync();

        bool key_found = tile.any(found);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("Post DupCheck", my_tid, found, key_found);
        // if (my_key ==2030567423 ) DEBUG_PI_TILE_BULK("MyKey", my_key, tile_id, key);
        // tile.sync();

        if (key_found)
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP FOUND", key, my_tid, tile_id);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("DUP MATCH", key, my_key, testkey);
            update_idx++;
            continue;
        }

        bool is_gt = active && (my_key > key);
        unsigned mask = tile.ballot(is_gt);

        if (mask != 0)
        {
            insert_index = __ffs(mask);
            testkey = tile.shfl(my_key, insert_index - 1);
            testoffset = tile.shfl(my_offset, insert_index - 1);
            tid_test_key = insert_index - 1;
        }
        else
        {
            insert_index = curr_size + 1;
            max_testkey = true;
            testkey = curr_max;
        }

        tile.sync();
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("InsertPos", insert_index, my_tid, is_gt);
        // if (tile_id == 1) DEBUG_PI_TILE_BULK("TestKey", testkey, curr_max, mask);

        if (my_tid == 0) // insert the first key
        {
            set_key_node<key_type>(curr_node, insert_index, key);
            set_offset_node<key_type>(curr_node, insert_index, offset);
            insert_count++;
            update_idx++;
            smallsize next_index = insert_index + 1;
        }
        tile.sync();
        next_index = tile.shfl(next_index, 0);
        update_idx = tile.shfl(update_idx, 0);
        insert_count = tile.shfl(insert_count, 0);
        key_type next_key = 0;
        bool need_new_testkey = false;

        while (update_idx <= maxindex && insert_count < free_space && next_index <= node_size)
        {
            next_key = update_list[update_idx];
            offset = offset_list[update_idx];
            // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", update_idx, key, testkey);
            // if (next_key ==1248854015 ) DEBUG_PI_TILE_BULK("INNER LOOP Key", next_key, my_tid, tile_id);

            if (next_key == testkey && !max_testkey)
            {
                set_key_node<key_type>(curr_node, next_index, testkey); // write the old values
                set_offset_node<key_type>(curr_node, next_index, testoffset);

                tid_test_key++;
                next_index++;
                update_idx++;
                need_new_testkey = true;
            }
            else if (next_key > testkey)
            {
                // new testkey - next one over
                if (max_testkey)
                    break; // means all keys in the node are less than next_key. next_key belongs in next node.
                set_key_node<key_type>(curr_node, next_index, testkey);
                set_offset_node<key_type>(curr_node, next_index, testoffset);
                next_index++;
                tid_test_key++;
                need_new_testkey = true;
                // use same update_idx key.
            }
            else
            {
                key = next_key;

                // if (next_key ==1248854015) DEBUG_PI_TILE_BULK(" Performing INsert", key, offset, next_index);

                set_key_node<key_type>(curr_node, next_index, key);
                set_offset_node<key_type>(curr_node, next_index, offset);

                // if (next_key ==1248854015) print_node<key_type>(curr_node, node_size);
                insert_count++;
                update_idx++;
                next_index++;
                need_new_testkey = false;
            }

            if (!need_new_testkey)
                continue; // back to top

            if (update_idx > maxindex || next_index > node_size || tid_test_key >= node_size)
                break; // no more space in the node

            next_key = update_list[update_idx];
            if (next_key > curr_max)
                break;

            if (my_tid >= tid_test_key)
            {
                found = (my_key == next_key);
                is_gt = active && (my_key > next_key);
                mask = tile.ballot(is_gt);
            }

            if found --figure out what to do :) \\

            if (mask != 0)
                {
                    insert_index = __ffs(mask);
                    testkey = tile.shfl(my_key, insert_index - 1);
                    testoffset = tile.shfl(my_offset, insert_index - 1);
                    tid_test_key = insert_index - 1;
                }
            else
            {
                insert_index = curr_size + insert_count;
                max_testkey = true;
                testkey = curr_max;
            }

            testkey = tile.shfl(my_key, tid_test_key);
            testoffset = tile.shfl(my_offset, tid_test_key);

            while (next_index <= node_size && testkey > 0 && next_key > testkey)
            {
                set_key_node<key_type>(curr_node, next_index, testkey);
                set_offset_node<key_type>(curr_node, next_index, testoffset);

                tid_test_key++;
                testkey = tile.shfl(my_key, tid_test_key);
                testoffset = tile.shfl(my_offset, tid_test_key);
                next_index++;
            }
            // if (update_idx > maxindex) break;
            //     next_key = update_list[update_idx];

            // index_test_key++;
            // continue;
        }
        // if (next_key ==1248854015) DEBUG_PI_TILE_BULK("InsertDone", insert_count, curr_size, next_index);

        tile.sync();

        insert_count = tile.shfl(insert_count, 0);
        update_idx = tile.shfl(update_idx, 0);
        total_insert_count += insert_count;
        curr_size += insert_count;
        if (my_tid == 0)
        {
            cg::set<smallsize>(curr_node, sizeof(key_type), curr_size);
            // if (tile_id == 1) print_node<key_type>(curr_node, node_size);
        }

        curr_size = tile.shfl(curr_size, 0);

        // HEREE
        //---->  update_more_keys<key_type>(tile, curr_node, next_index, free_space, insert_index, insert_count, my_key, my_offset, curr_size, node_size);

        // if (tile_id == 1) DEBUG_PI_TILE_BULK("SizePost", my_tid, curr_size, insert_count);

        if (active && my_tid >= (insert_index - 1))
        {
            //  if (tile_id == 1) DEBUG_PI_TILE_BULK("Shift", my_tid, insert_index, node_size);
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("ShiftVal", my_key, my_offset, curr_size);

            set_key_node<key_type>(curr_node, my_tid + 1 + insert_count, my_key);
            set_offset_node<key_type>(curr_node, my_tid + 1 + insert_count, my_offset);
        }
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("Post Full Update to Node", my_tid, curr_size, insert_count);
        // if (my_tid==0 && tile_id == 1) print_node<key_type>(curr_node, node_size);

        tile.sync();

        // if (curr_size > node_size || insert_count > free_space)
        // {
        //     ERROR_INSERTS("Overflow", key, curr_size, node_size);
        //}

        if (curr_size == node_size && update_idx < maxindex &&
            (total_keys - total_insert_count > 0) &&
            (update_list[update_idx] < curr_max))
        {
            // if (tile_id == 1) DEBUG_PI_TILE_BULK("SplitTrigger", curr_size, update_list[update_idx], curr_max);
            process_split_tile<key_type>(tile, curr_node, launch_params);
            // extract the new node max again
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // tile.sync();
        // if (my_tid==0 && tile_id == 1) DEBUG_PI_TILE_BULK("END ONE ROUND", my_tid, tile_id);
    }

#ifdef PRINT_PROCESS_INSERTS_END
    __syncthreads();
    if (my_tid == 0 && tile_id == 1)
    {
        printf("END INSERTIONS: PRINT ALL NODES\n");
        print_set_nodes_and_links<key_type>(launch_params, my_tid);
    }
#endif
}

*/ 