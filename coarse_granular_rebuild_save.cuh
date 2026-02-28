#ifndef INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_REBUILD_CUH
#define INDEX_PROTOTYPE_IMPL_COARSE_GRANULAR_REBUILD_CUH

#include "coarse_granular_inserts.cuh"

#include <cuda_runtime.h>
#include <iostream>
//#include "cuda_buffer.h"

//#include "impl_cg_rtx_index_updates.cuh"

//template <typename key_type>
//void find_maxvalues(cuda_buffer<key_type> &maxvalues_buffer, cuda_buffer<uint8_t> &ordered_node_pairs_buffer, size_t partition_count_with_overflow, smallsize node_stride);

template <typename key_type>
GLOBALQUALIFIER void count_nodes_per_bucket_kernel(
    const void *node_buffer,
    const void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,
    smallsize partition_count_with_overflow,
    smallsize *nodes_per_bucket)
{
    // Get thread ID
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread ID is within valid bounds
    if (tid >= partition_count_with_overflow)
        return;

    // Start at the first node for this partition
    const uint8_t *curr_node = reinterpret_cast<const uint8_t *>(node_buffer) + tid * node_stride;
    
    // Initialize count
    size_t count = 1;  // At least the first node is counted

    // Get the last position offset
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);
    smallsize next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);

    // Traverse linked nodes
    while (next_offset != 0)
    {
        next_offset--; // Adjust for +1 offset in storage

        // Safety check: Ensure the offset is within bounds
        if (next_offset >= allocation_buffer_count)
        {
            printf("ERROR: NEXT OFFSET EXCEEDS ALLOCATION BUFFER COUNT\n");
            break;
        }

        // Move to the next linked node
        curr_node = reinterpret_cast<const uint8_t *>(allocation_buffer) + next_offset * node_stride;
        count++;

        // Extract the next offset
        next_offset = cg::extract<smallsize>(curr_node, lastposition_bytes);
    }

    // Store the final count in nodes_per_bucket
    nodes_per_bucket[tid] = count;
}
template <typename key_type>
void launch_count_nodes_per_bucket(
    const void *node_buffer,
    const void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize allocation_buffer_count,  // Make sure this is the right count
    smallsize partition_count_with_overflow,
    smallsize *nodes_per_bucket)
{
    // Allocate GPU buffer for results using cuda_buffer
    cuda_buffer<smallsize> d_nodes_per_bucket;
    d_nodes_per_bucket.alloc(partition_count_with_overflow);
    d_nodes_per_bucket.zero(); // Initialize all counts to zero

    // Define block and grid sizes using MAXBLOCKSIZE
    int threadsPerBlock = MAXBLOCKSIZE;
    int numBlocks = (partition_count_with_overflow + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    count_nodes_per_bucket_kernel<key_type><<<numBlocks, threadsPerBlock>>>(
        node_buffer, allocation_buffer, node_size, node_stride,
        allocation_buffer_count,  // Ensure this is the correct count
        partition_count_with_overflow, d_nodes_per_bucket.ptr());

    // Copy results back to host
    cudaStreamSynchronize(0);
    CUERR;

    d_nodes_per_bucket.download(nodes_per_bucket, partition_count_with_overflow);

    // GPU memory is automatically freed in the destructor of cuda_buffer
}
template <typename key_type>
GLOBALQUALIFIER void rebuild_kernel(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    smallsize partition_count_with_overflow,
    smallsize total_nodes_used_from_AR,
    smallsize total_nodes,
    smallsize *nodes_per_bucket,
    smallsize *prefix_sum_array)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= partition_count_with_overflow)
        return;

    smallsize node_count_for_this_partition = nodes_per_bucket[tid];
    smallsize prefix_sum_value = (tid ==0)? 0 : prefix_sum_array[tid-1];
    smallsize lastposition_bytes = get_lastposition_bytes<key_type>(node_size);

#ifdef PRINT_REBUILD_DATA

    //printf("Thread %u: nodes_per_bucket[%u] = %u, Actual prefix_sum_array[%u] = %u  and prefix_sum_value %u \n", tid, tid, nodes_per_bucket[tid], tid, prefix_sum_array[tid], prefix_sum_value);
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Before REBUILD Total # Nodes: %u\n", partition_count_with_overflow) ;
        print_set_nodes_and_links<key_type>(node_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, total_nodes_used_from_AR);
    }
#endif
    // Copying work to new Rep Buffer
      // Get the correct location in node_buffer and correct location to copy into rep_buffer 
        //uint8_t *rep_node = reinterpret_cast<uint8_t *>(representative_temp_buffer) + (prefix_sum_value + i) * node_stride;
    auto curr_node = reinterpret_cast<uint8_t *>(node_buffer) + tid * node_stride;

    //extract size from curr_node
    smallsize size_currnode = cg::extract<smallsize>(curr_node, sizeof(key_type));

    // to compute the  copy location in the representative buffer we need to take prefix_sum_value--
    //prefix_sum_value--; // Adjust for +1 offset in storage   
for (smallsize i = 0; i < node_count_for_this_partition; i++)
    {
        auto rep_node = reinterpret_cast<uint8_t *>(representative_temp_buffer) + (prefix_sum_value+i) * node_stride;
   
        smallsize r = 0;
        for (smallsize m = 0; m <= node_size; m++) {  
            key_type key = extract_key_node<key_type>(curr_node, m);
            smallsize offset = extract_offset_node<key_type>(curr_node, m);

            if (key == tombstone && offset == 0) {
                // Skip incrementing m, effectively "retrying" same index
                continue;
            }

            //set_key_node<key_type>(rep_node, m, key);
            //set_offset_node<key_type>(rep_node, m, offset);
            set_key_node<key_type>(rep_node, r, key);
            set_offset_node<key_type>(rep_node, r, offset);
            r++;

            //m++;  // Increment manually
        }

        // Check that the link ptr should already be 0 
         smallsize linked_ptr = cg::extract<smallsize>(rep_node, lastposition_bytes);
        if (linked_ptr != 0)
        {
            printf("ERROR REBUILD: New Rep NODE: Tid: %d Linked Ptr is Non Zero\n", tid);
            print_node<key_type>(rep_node, node_size);
        }

        // Move to the next node
        smallsize linked_ptr_curr_node = cg::extract<smallsize>(curr_node, lastposition_bytes);
        if (linked_ptr_curr_node == 0)
        {   
           // printf("ERROR REBUILD: Current NODE: Tid: %d Linked Ptr is Zero\n", tid);
            //print_node<key_type>(curr_node, node_size);
            break;
        }
        linked_ptr_curr_node--; // Adjust for +1 offset in storage
        curr_node = reinterpret_cast<uint8_t *>(allocation_buffer) + linked_ptr_curr_node * node_stride;
        //rep_node += node_stride;    
        //auto rep_node = reinterpret_cast<uint8_t *>(representative_temp_buffer) + (prefix_sum_value) * node_stride;
        //auto curr_node = reinterpret_cast<uint8_t *>(node_buffer) + tid * node_stride;



    }
#ifdef PRINT_REBUILD_DATA
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("After REBUILD Total # Nodes: %u\n", total_nodes);
        print_set_nodes_and_links<key_type>(representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, total_nodes_used_from_AR);
    }
#endif
}

GLOBALQUALIFIER void compute_prefix_sum_array(smallsize *d_nodes_per_bucket, smallsize *d_prefix_sum, smallsize size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_prefix_sum[0] = d_nodes_per_bucket[0];
       // DEBUG_REBUILD("Prefix Sum Array First Element", d_prefix_sum[0], d_nodes_per_bucket[0], size);
        for (smallsize i = 1; i < size; i++) {
            d_prefix_sum[i] = d_prefix_sum[i - 1] + d_nodes_per_bucket[i];
        }
    }
}

void compute_prefix_sum_on_gpu(cuda_buffer<smallsize> &nodes_per_bucket, cuda_buffer<smallsize> &prefix_sum_array, smallsize partition_count_with_overflow) {
    prefix_sum_array.resize(partition_count_with_overflow);
    compute_prefix_sum_array<<<1, 1>>>(nodes_per_bucket.ptr(), prefix_sum_array.ptr(), partition_count_with_overflow);
    cudaStreamSynchronize(0);
}

template <typename key_type>
void rebuild_gpu_structures(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    key_type *maxbuf,
    smallsize node_size,
    smallsize node_stride,
    double *time_ms,
    smallsize total_nodes_used_from_AR,
    smallsize partition_count_with_overflow)
{
    smallsize node_count = total_nodes_used_from_AR + partition_count_with_overflow;
    smallsize *nodes_per_bucket = new smallsize[partition_count_with_overflow];

    launch_count_nodes_per_bucket<key_type>(
        node_buffer, allocation_buffer, node_size, node_stride,
        total_nodes_used_from_AR, partition_count_with_overflow, nodes_per_bucket);

    cuda_buffer<smallsize> d_nodes_per_bucket;
    d_nodes_per_bucket.alloc_and_upload(std::vector<smallsize>(nodes_per_bucket, nodes_per_bucket + partition_count_with_overflow));

    // Compute PREFIX SUM ARRAY
    cuda_buffer<smallsize> d_prefix_sum_array;
    compute_prefix_sum_on_gpu(d_nodes_per_bucket, d_prefix_sum_array, partition_count_with_overflow);

    rebuild_kernel<key_type><<<SDIV(partition_count_with_overflow, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
        node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride,
        partition_count_with_overflow, total_nodes_used_from_AR, node_count, d_nodes_per_bucket.ptr(), d_prefix_sum_array.ptr());

    cudaStreamSynchronize(0);
    CUERR;
    delete[] nodes_per_bucket;

    // Free CUDA buffers
    d_nodes_per_bucket.free();
    d_prefix_sum_array.free();

    //const key_type *maxbuff = static_cast<const key_type *>(maxbuf);


    // Print all max values in the maxbuf
   // printf("PRINTING MAX VALUES at the end of rebuild_gpu_structures\n");
   // for (int i = 0; i < partition_count_with_overflow; i++) {
   //     printf("maxbuff[%d] %llu\n", i, static_cast<unsigned long long>(maxbuff[i]));
   // }
}


//--------------------------------- OLD KERNELS ---------------------------------

template <typename key_type>
GLOBALQUALIFIER void rebuild_kernel_old(void *node_buffer, void *representative_temp_buffer, void *allocation_buffer, smallsize node_size, smallsize node_stride, smallsize partition_count_with_overflow, smallsize total_nodes)
{

    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= total_nodes)
        return;

/* 
    smallsize total_nodes = partition_count_with_overflow - 1; // total nodes not including the overflow node
    smallsize key_offset_bytes = compute_key_offset_bytes<key_type>();
    smallsize my_node = tid / initial_num_keys_each_node; // which node this tid belongs to
    smallsize my_place_in_node = (tid + 1) % initial_num_keys_each_node;

    smallsize byte_offset = (node_stride * my_node) + ((tid % initial_num_keys_each_node) + 1) * key_offset_bytes;

    cg::set<key_type>(node_buffer, byte_offset, keys[tid]);
    cg::set<smallsize>(node_buffer, byte_offset + sizeof(key_type), offsets[tid]);

    // extra work if the thread is the last member in the group
    // insert maxkey and size at the front of the node
    if (my_place_in_node == 0)
    {
        // insert maxkey
        cg::set<key_type>(node_buffer, my_node * node_stride, keys[tid]);
        // insert size
        cg::set<smallsize>(node_buffer, my_node * node_stride + sizeof(key_type), initial_num_keys_each_node);
        // clear the lastptr to 0
        cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
    }

    if (my_node == total_nodes - 1)
    { // last node not including overflow node

        smallsize remainder = key_count % initial_num_keys_each_node;
        bool has_remainder = (key_count % initial_num_keys_each_node != 0);
        if (has_remainder)
            my_place_in_node = ((tid + 1) % remainder);

        if (my_place_in_node == 0)
        {

            // this code block repeated in case there is a remainder. last node max needs to be set
            if (has_remainder)
            { // update size correctly in last actual node (not overflow node)

                DEBUG_BUILD("Node Layout Kernel: Has Remainder true", 5, tid, remainder, my_node, my_place_in_node, keys[key_count - 1]);

                // set the size to be remainder and the max_key to be the last key in the dataset
                cg::set<smallsize>(node_buffer, (total_nodes - 1) * node_stride + sizeof(key_type), remainder);
                cg::set<key_type>(node_buffer, my_node * node_stride, keys[key_count - 1]);
                // clear the lastptr to 0
                cg::set<smallsize>(node_buffer, my_node * node_stride + (node_size + 1) * key_offset_bytes, 0);
            }

            key_type datamax; // datamax is the max value for the overflow node
            if (sizeof(key_type) == 8)
            {
                datamax = static_cast<key_type>(std::numeric_limits<key64>::max());
            }
            else if (sizeof(key_type) == 4)
            {
                datamax = static_cast<key_type>(std::numeric_limits<key32>::max());
            }

            // set max for OVERFLOW NODE ONLY (overflow node size is 0 at initial build time)
            cg::set<key_type>(node_buffer, (total_nodes)*node_stride, datamax);
        }
    }



template <typename key_type>
void rebuild_gpu_structures_A(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    size_t node_size,
    size_t node_stride,
    double *time_ms,
    size_t total_nodes_used_from_AR,
    size_t partition_count_with_overflow)
{
   // scoped_cuda_timer timer(0, time_ms);
   size_t node_count = total_nodes_used_from_AR +partition_count_with_overflow;
   size_t nodes_per_bucket[partition_count_with_overflow];

///--> HERE   kernel to compute the nodes per bucket



   rebuild_kernel<key_type><<<SDIV(node_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, node_count);
   cudaStreamSynchronize(0);
   CUERR
   // cudaStreamSynchronize(0);
    //CUERR
}



template <typename key_type>
void rebuild_gpu_structures(
    void *node_buffer,
    void *representative_temp_buffer,
    void *allocation_buffer,
    smallsize node_size,
    smallsize node_stride,
    double *time_ms,
    smallsize total_nodes_used_from_AR,
    smallsize partition_count_with_overflow)
{
    // Start GPU timer
    // scoped_cuda_timer timer(0, time_ms);

    smallsize node_count = total_nodes_used_from_AR + partition_count_with_overflow;

    // Allocate array for nodes per bucket on host
    smallsize *nodes_per_bucket = new smallsize[partition_count_with_overflow];

    /// ---> CALL THE KERNEL TO COMPUTE THE NODES PER BUCKET
    launch_count_nodes_per_bucket<key_type>(
        node_buffer, allocation_buffer, node_size, node_stride,
        total_nodes_used_from_AR, partition_count_with_overflow, nodes_per_bucket);

    // Allocate nodes_per_bucket on GPU using cuda_buffer
    cuda_buffer<smallsize> d_nodes_per_bucket;
    d_nodes_per_bucket.alloc_and_upload(std::vector<smallsize>(nodes_per_bucket, nodes_per_bucket + partition_count_with_overflow));


   // ----->> HERE  to do PREFIX SUM ARRAY compUTATION

    // Launch the rebuild kernel, passing nodes_per_bucket
   //---> for now Just One Thread per bucket
      // rebuild_kernel<key_type><<<SDIV(node_count, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride, partition_count_with_overflow, node_count, d_nodes_per_bucket.ptr());

        rebuild_kernel<key_type><<<SDIV(partition_count_with_overflow, MAXBLOCKSIZE), MAXBLOCKSIZE>>>(
        node_buffer, representative_temp_buffer, allocation_buffer, node_size, node_stride,
        partition_count_with_overflow, node_count, d_nodes_per_bucket.ptr());

    // Synchronize kernel execution
    cudaStreamSynchronize(0);
    CUERR;

    // Cleanup host memory
    delete[] nodes_per_bucket;

    // cudaStreamSynchronize(0);
    // CUERR;
}


for (smallsize m = 0; m <= node_size; ) {  // No ++m in loop header
    key_type key = extract_key_node<key_type>(curr_node, m);
    smallsize offset = extract_offset_node<key_type>(curr_node, m);

    if (key == 1 && offset == 0) {
        // Skip incrementing m, effectively "retrying" same index
        continue;
    }

    set_key_node<key_type>(rep_node, m, key);
    set_offset_node<key_type>(rep_node, m, offset);

    m++;  // Increment manually
}

    */
}

#endif

/*

 // Copy the node
        //we copy the full node_size bc we should have tombstone present
        smallsize c ,r = 0;
        while( c <= node_size)
        {
            key_type key = extract_key_node<key_type>(curr_node, c);
            smallsize offset = extract_offset_node<key_type>(curr_node, c);
            // check for tombstones and ignore. size of node should still be accurate
            // we are removing all unnecessary tombstones
            if (key == 1 && offset == 0){
                c++;                
                continue;
            }
                
            set_key_node<key_type>(rep_node, r, key);
            set_offset_node<key_type>(rep_node, r, offset);
            c++; r++;
        }

        */