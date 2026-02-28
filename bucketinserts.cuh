
// new Experiments Jan 2025

//--------------- Node_Split
template <typename key_type>
DEVICEQUALIFIER void split_node(void *curr_node, key_type *keys_array, smallsize *offsets_array, smallsize num_elements, smallsize node_size, updatable_cg_params *launch_params, int tid)
{
    // Read necessary parameters from launch_params
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize node_stride = launch_params->node_stride;
    smallsize free_index = launch_params->free_node;
    smallsize partition_size = launch_params->partition_size;
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;

    if (free_index >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    // Increment the free_node index atomically
    smallsize free_value = atomicAdd((&launch_params->free_node), 1ULL);
    //printf("Tid: %d, Free Value: %d\n", tid, free_value);

    auto linked_node = reinterpret_cast<uint8_t *>(allocation_buffer) + node_stride * free_value;

    /******* Check for available space in Allocation Buffer **********/
    if (free_value >= allocation_buffer_count)
    {
        ERROR_INSERTS("ERROR: Node Split: Allocation Buffer is Full:", tid, free_index);
        return;
    }

    //printf("Before SPLIT loop: Tid: %d, Print Both Nodes\n", tid);
    //print_node<key_type>(curr_node, partition_size);
    //printf("tid: %d Linked Node\n", tid);
    //print_node<key_type>(linked_node, node_size);

    // Split the keys and offsets between curr_node and linked_node
    smallsize mid = num_elements / 2;
    for (smallsize i = 0; i < mid; ++i)
    {
        set_key_node<key_type>(curr_node, i + 1, keys_array[i]);
        set_offset_node<key_type>(curr_node, i + 1, offsets_array[i]);
    }
    for (smallsize i = mid; i < num_elements; ++i)
    {
        set_key_node<key_type>(linked_node, i - mid + 1, keys_array[i]);
        set_offset_node<key_type>(linked_node, i - mid + 1, offsets_array[i]);
        set_key_node<key_type>(curr_node, i + 1, 0);
        set_offset_node<key_type>(curr_node, i + 1, 0);
    }

    //printf("CHECK: SPLIT NODES: Tid: \n", tid);
    //print_node<key_type>(curr_node, partition_size);
    //print_node<key_type>(linked_node, node_size);

    // #ifdef ERROR_CHECK
    //  Check position partition_size+1 in both nodes, it should be zero
    key_type test_curr_key = extract_key_node<key_type>(curr_node, partition_size + 1);
    key_type test_linked_key = extract_key_node<key_type>(linked_node, partition_size + 1);
    if (test_curr_key != 0 || test_linked_key != 0)
    {
        printf("ERROR: SPLIT NODE NON ZERO: Tid: %d, Curr_Node || Linked Node [partition_size +1] Non Zero\n", tid);
        print_node<key_type>(curr_node, node_size);
        print_node<key_type>(linked_node, node_size);
    }
    // #endif

#ifdef NODESPLIT
    printf("AFTER Split: Tid: %d, After Copy Half Curr Node to Linked Node\n", tid);
    print_node<key_type>(curr_node, node_size);
    print_node<key_type>(linked_node, node_size);
#endif

    // Set sizes and max values for curr_node and linked_node
    cg::set<smallsize>(curr_node, sizeof(key_type), mid);
    cg::set<smallsize>(linked_node, sizeof(key_type), num_elements - mid);

    // Set max values for curr_node and linked_node
    key_type original_max = cg::extract<key_type>(curr_node, 0);
    key_type curr_max = extract_key_node<key_type>(curr_node, mid);
    cg::set<key_type>(curr_node, 0, curr_max);
    cg::set<key_type>(linked_node, 0, original_max);

    // #ifdef NODESPLIT
    DEBUG_PI_BUCKET("After Set Maxes and Sizes", curr_max, original_max);
    //print_node<key_type>(curr_node, node_size);
    //print_node<key_type>(linked_node, node_size);
    // #endif

    // Set linked pointer
    smallsize last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
    if (last_position_value != 0)
    {
        cg::set<smallsize>(linked_node, lastposition_bytes, last_position_value);
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert at plus 1 to avoid 0
    }
    else
    {
        cg::set<smallsize>(curr_node, lastposition_bytes, free_value + 1); // insert free_value+1 into last position since we do not allow 0 in the last position
        smallsize linked_ptr = cg::extract<smallsize>(linked_node, lastposition_bytes);
        // #ifdef NODESPLIT
        if (linked_ptr != 0)
        {
            printf("ERROR: SPLIT NODE: Tid: %d Linked Ptr is Non Zero\n", tid);
            print_node<key_type>(linked_node, node_size);
        }
        // #endif
    }

    DEBUG_PI_BUCKET("After Links Set", tid, last_position_value);
    //print_node<key_type>(curr_node, node_size);
    //print_node<key_type>(linked_node, node_size);
}
//-------------------------------

template <typename key_type>
DEVICEQUALIFIER void copy_arrays_to_node(void *curr_node, key_type *keys_array, smallsize *offsets_array, smallsize num_elements, smallsize node_size, updatable_cg_params *launch_params, int tid)
{
    // check if a split is requried. If so, split the node into two nodes and place keys in the correct node
    //  Copy all the keys and offsets from arrays into the curr_node and next_node pointed to by curr_node
    //  Update the size of curr_node
    // cg::set<smallsize>(curr_node, sizeof(key_type), num_elements);  //update the size
    //  size will be correct set in the split function
    DEBUG_PI_BUCKET("IN Copy Function ", tid, num_elements, node_size);

    if (num_elements >= node_size)
    {
        //printf("Tid: %d, Copy Arrays to Node:  Splitting Node\n", tid);
        // print value in keys_array and offsets_array side by side on one line, and then print the curr_node using print_node function
        for (smallsize i = 0; i < num_elements; ++i)
        {
            DEBUG_PI_BUCKET("keys/offsets ", tid, keys_array[i], offsets_array[i]); // %u\n", tid, i, keys_array[i], i, offsets_array[i]);
        }
        //print_node<key_type>(curr_node, node_size);

        split_node<key_type>(curr_node, keys_array, offsets_array, num_elements, node_size, launch_params, tid);
        return;
    }

    for (smallsize i = 0; i < num_elements; ++i)
    {   DEBUG_PI_BUCKET("Just copying: keys/offsets ", tid, keys_array[i], offsets_array[i]); // %u\n", tid, i, keys_array[i], i, offsets_array[i]);

        set_key_node<key_type>(curr_node, i + 1, keys_array[i]);
        set_offset_node<key_type>(curr_node, i + 1, offsets_array[i]);
    }
    // Update the size of curr_node
    cg::set<smallsize>(curr_node, sizeof(key_type), num_elements); // update the size
}

template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    // Compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize returnval = 0; // 0 not inserted was present
                             // 1 successfully inserted
                             // 2 error of some type occurred
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    auto buf = launch_params->ordered_node_pairs;
    smallsize maxkeys_pernode = node_size; // allow 2xpartionsize keys per node
                                           // last index is used for next pointer

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

#ifdef PRINT_PROCESS_INSERTS
    if (tid == 0)
    {
        DEBUG_PI_BUCKET("IN Process Inserts: Tid: ", tid, minindex, maxindex);
        print_set_nodes<key_type>(launch_params, tid);
    }
#endif

    // Create two arrays of size node_size
    key_type *keys_array = new key_type[node_size];
    smallsize *offsets_array = new smallsize[node_size];
    smallsize k = minindex;

    while (k <= maxindex)
    {

        key_type next_insert_key = update_list[k];

        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        DEBUG_PI_BUCKET("Process Inserts per Bucket", tid, curr_max, next_insert_key);
        while (curr_max < next_insert_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
#ifdef PRINT_PROCESS_INSERTS
            DEBUG_PI_BUCKET("Find the correct Node in Bucket", tid, curr_max, next_insert_key);
            print_node<key_type>(curr_node, node_size);
#endif

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // clear the two arrays
        // for (smallsize i = 0; i < node_size; ++i)
        // {
        //    keys_array[i] = 0;
        //    offsets_array[i] = 0;
        // }

        // Clear the arrays
        memset(keys_array, 0, node_size * sizeof(key_type));
        memset(offsets_array, 0, node_size * sizeof(smallsize));

        // Extract the current size of keys in curr_node
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        // Copy all the keys and offsets from curr_node into the arrays
        smallsize j = 0;            // Index for keys_array and offsets_array
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);
            DEBUG_PI_BUCKET("Top For Loop Process_Inserts per Bucket", tid, curr_key, curr_offset);

            // Insert all keys from update_list that are less than the current key
            while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key)
            {

                keys_array[j] = update_list[update_idx];
                offsets_array[j] = offset_list[update_idx];
                DEBUG_PI_BUCKET("val in update_list is less than curr_key", tid, curr_key, curr_offset);
                DEBUG_PI_BUCKET("Inserted val from update_list bc it was less than curr_key", tid, update_list[update_idx], offset_list[update_idx]);
                j++;
                update_idx++;
                insert_count++;
            }

            // If the current key matches a key in update_list, update the offset
            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];
                DEBUG_PI_BUCKET("Update offset", tid, curr_key, curr_offset);
                update_idx++;
            }

            keys_array[j] = curr_key;
            offsets_array[j] = curr_offset;
            DEBUG_PI_BUCKET("Inserted from curr_node", tid, curr_key, curr_offset);
            j++;
        }

        // Insert any remaining keys from update_list
        while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max)
        {
            // if (insert_count + curr_size >= node_size)
            //{

            //  break;
            //  }
            keys_array[j] = update_list[update_idx];
            offsets_array[j] = offset_list[update_idx];
            DEBUG_PI_BUCKET("Insert remaining from update_list", tid, update_list[update_idx], curr_max);
            DEBUG_PI_BUCKET("Insert remaining ", j, keys_array[j], offsets_array[j]);
            j++;
            update_idx++;
            insert_count++;
        }

        // Copy the arrays back to curr_node
        // also perform a split if required
        copy_arrays_to_node<key_type>(curr_node, keys_array, offsets_array, j, node_size, launch_params, tid);
        DEBUG_PI_BUCKET("size of currnode", tid, curr_size, j);

        DEBUG_PI_BUCKET("After Copying Arrays to Node: PRINT NODE", tid);
       // if (tid == 0)
            //print_node<key_type>(curr_node, node_size);

        k = update_idx;

    } // end loop for all insertinos from the update_list
    // Clean up
    delete[] keys_array;
    delete[] offsets_array;
    
//#define DEBUG_SELECTIVE
    #ifdef DEBUG_SELECTIVE
    __syncthreads(); // Synchronize all threads in the block
    if (tid == 4)
    {
        printf("END INSERTIONS PRINT ALL NODES\n    ");
        print_set_nodes<key_type>(launch_params, tid);
    }
    #endif
}



------------ NO DYNMAIC ALLOCATION


template <typename key_type>
DEVICEQUALIFIER void process_inserts_per_bucket_FIRST(key_type maxkey, smallsize minindex, smallsize maxindex, updatable_cg_params *launch_params, void *curr_node, key_type num_elements, int tid)
{
    // Compute tid here
    smallsize idx = blockIdx.x * blockDim.x + threadIdx.x;
    smallsize totalstoredsize = launch_params->stored_size;
    void *allocation_buffer = launch_params->allocation_buffer;
    smallsize partition_size = launch_params->partition_size;
    smallsize node_size = launch_params->node_size;
    smallsize partition_count_with_overflow = launch_params->partition_count_with_overflow;
    smallsize node_stride = launch_params->node_stride;
    smallsize allocation_buffer_count = launch_params->allocation_buffer_count;
    smallsize returnval = 0; // 0 not inserted was present
                             // 1 successfully inserted
                             // 2 error of some type occurred
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    auto buf = launch_params->ordered_node_pairs;
    smallsize maxkeys_pernode = node_size; // allow 2xpartition_size keys per node
                                           // last index is used for next pointer

    const key_type *update_list = static_cast<const key_type *>(launch_params->update_list);
    const smallsize *offset_list = static_cast<const smallsize *>(launch_params->offset_list);

    // New: Use thread-specific sections of copy_update_list and copy_offset_list
    key_type *copy_update_list = static_cast<key_type *>(launch_params->copy_update_list);
    smallsize *copy_offset_list = static_cast<smallsize *>(launch_params->copy_offset_list);

    key_type *index_in_copy_updates = copy_update_list + tid * node_size;
    smallsize *index_in_copy_offsets = copy_offset_list + tid * node_size;

    // Initialize the memory for this thread's section
    memset(index_in_copy_updates, 0, node_size * sizeof(key_type));
    memset(index_in_copy_offsets, 0, node_size * sizeof(smallsize));

    #ifdef PRINT_PROCESS_INSERTS
    if (tid == 0)
    {
        DEBUG_PI_BUCKET("IN Process Inserts: Tid: ", tid, minindex, maxindex);
        print_set_nodes<key_type>(launch_params, tid);
    }
    #endif
    DEBUG_PI_BUCKET("IN Process Inserts BUCKETS: Tid: ", tid, minindex, maxindex);

    smallsize k = minindex;

    while (k <= maxindex)
    {
        key_type next_insert_key = update_list[k];

        smallsize last_position_value = 0; // extract_key_node<key_type>(curr_node, lastpositionptr);
        key_type curr_max = cg::extract<key_type>(curr_node, 0);

        //DEBUG_PI_BUCKET("While Loop Process Inserts per Bucket", tid, curr_max, next_insert_key);
        while (curr_max < next_insert_key)
        {
            last_position_value = cg::extract<smallsize>(curr_node, lastposition_bytes);
            last_position_value--; // decrement b/c it is inserted into node as +1 to avoid 0
            #ifdef PRINT_PROCESS_INSERTS
            DEBUG_PI_BUCKET("Find the correct Node in Bucket", tid, curr_max, next_insert_key);
            print_node<key_type>(curr_node, node_size);
            #endif

            if (last_position_value >= allocation_buffer_count)
            {
                ERROR_INSERTS("ERROR: LAST PTR EXCEEDS NUM PARTITIONS", tid, curr_max, partition_count_with_overflow);
                return;
            }
            curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + last_position_value * node_stride;
            curr_max = cg::extract<key_type>(curr_node, 0);
        }

        // Extract the current size of keys in curr_node
        smallsize curr_size = cg::extract<smallsize>(curr_node, sizeof(key_type));

        smallsize j = 0;            // Index for keys and offsets
        smallsize update_idx = k;   // Index for update_list
        smallsize insert_count = 0; // Counter for new insertions

        for (smallsize i = 1; i <= curr_size; ++i)
        {
            key_type curr_key = extract_key_node<key_type>(curr_node, i);
            smallsize curr_offset = extract_offset_node<key_type>(curr_node, i);
            //DEBUG_PI_BUCKET(" Top For Loop Process_Inserts per Bucket ", tid, curr_key, curr_offset);

            while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] < curr_key)
            {
                index_in_copy_updates[j] = update_list[update_idx];
                index_in_copy_offsets[j] = offset_list[update_idx];
               // if (tid ==14) DEBUG_PI_BUCKET(" val in update_list is less than curr_key ", tid, curr_key, curr_offset);
               // if (tid ==14) DEBUG_PI_BUCKET(" Inserted val from update_list bc it was less than curr_key ", tid, update_list[update_idx], offset_list[update_idx]);
                j++;
                update_idx++;
                insert_count++;

                //print the values in index_in_copy_updates and index_in_copy_offsets that was just inserted
                //if (tid == 14) DEBUG_PI_BUCKET("Inserted from update_list", tid, update_list[update_idx], offset_list[update_idx]);
               // if (tid == 14) DEBUG_PI_BUCKET(" Inserted from copy lists", tid, index_in_copy_updates[j], index_in_copy_offsets[j] );

            }

            if (update_idx <= maxindex && update_list[update_idx] == curr_key)
            {
                curr_offset = offset_list[update_idx];
              //  if (tid == 14) DEBUG_PI_BUCKET("Update offset", tid, curr_key, curr_offset);
                update_idx++;
            }

            index_in_copy_updates[j] = curr_key;
            index_in_copy_offsets[j] = curr_offset;
          //  if (tid == 14) DEBUG_PI_BUCKET( "Inserted from curr_node", tid, curr_key, curr_offset );
           // if (tid == 14) DEBUG_PI_BUCKET( "Inserted to copy list normal curr_node values", tid, index_in_copy_updates[j], index_in_copy_offsets[j] );


            j++;
        }

        while ((insert_count + curr_size < node_size) && update_idx <= maxindex && update_list[update_idx] <= curr_max)
        {
            index_in_copy_updates[j] = update_list[update_idx];
            index_in_copy_offsets[j] = offset_list[update_idx];
           // if (tid == 14) DEBUG_PI_BUCKET(" Insert remaining from update_list ", tid, update_list[update_idx], curr_max);
            //if (tid == 14) DEBUG_PI_BUCKET(" Insert remaining ", j, index_in_copy_updates[j], index_in_copy_offsets[j]);
            //if (tid == 14) DEBUG_PI_BUCKET(" Inserted remaining to copy lists ", tid, index_in_copy_updates[j], index_in_copy_offsets[j]);

            j++;
            update_idx++;
            insert_count++;
        }

        if (tid == 14) {
            printf(" going to do COPY Tid: %d, Insert Count: %d, Curr Size: %d, Node Size: %d\n", tid, insert_count, curr_size, node_size);
            print_node<key_type>(curr_node, node_size);
            //print all keys in both copy lists, key and offsets on one line
            for (smallsize i = 0; i < node_size; ++i)
            {
                DEBUG_PI_BUCKET(" TID 14 keys/offsets ", tid, index_in_copy_updates[i], index_in_copy_offsets[i]);
            }

        }

         


          if (tid == 5) {
            printf(" going to do COPY Tid: %d, Insert Count: %d, Curr Size: %d, Node Size: %d\n", tid, insert_count, curr_size, node_size);
            print_node<key_type>(curr_node, node_size);
            //print all keys in both copy lists, key and offsets on one line
            for (smallsize i = 0; i < node_size; ++i)
            {
                DEBUG_PI_BUCKET(" TID 5 keys/offsets ", tid, index_in_copy_updates[i], index_in_copy_offsets[i]);
            }

           }
        copy_arrays_to_node<key_type>(curr_node, index_in_copy_updates, index_in_copy_offsets, j, node_size, launch_params, tid);
        //DEBUG_PI_BUCKET("size of currnode", tid, curr_size, j);

       // DEBUG_PI_BUCKET("After Copying Arrays to Node: PRINT NODE", tid);
       // if (tid == 5) print_node<key_type>(curr_node, node_size);
       // if (tid == 14) print_node<key_type>(curr_node, node_size);

        k = update_idx;
    }

//#define DEBUG_SELECTIVE
    #ifdef DEBUG_SELECTIVE
    __syncthreads(); // Synchronize all threads in the block
    if (tid == 14)
    {
        printf("END INSERTIONS PRINT ALL NODES\n    ");
        print_set_nodes<key_type>(launch_params, tid);
    }
    #endif
}