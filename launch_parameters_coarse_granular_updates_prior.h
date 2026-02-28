#ifndef LAUNCH_PARAMETERS_COARSE_GRANULAR_UPDATES_H
#define LAUNCH_PARAMETERS_COARSE_GRANULAR_UPDATES_H

#include "definitions.cuh"


template<typename key_type>
struct updatable_coarse_granular_launch_parameters {
    OptixTraversableHandle traversable;

    void* ordered_node_pairs;
    void* allocation_buffer;
   
    const key_type*  maxvalues;
    smallsize node_stride;
    const smallsize*  offset_list;
    const key_type*  update_list;

    smallsize update_size;
    bool deletions;
    smallsize free_node;
    smallsize partition_count_with_overflow;
    smallsize allocation_buffer_count;

    bool long_keys;
    bool aggregate_results;

    smallsize stored_size;
    smallsize partition_size;
    smallsize node_size;
    smallsize partition_count;

    // either 32-bit or 64-bit keys, use void if irrelevant
    const key_type* query_lower;
    const key_type* query_upper;

    smallsize* result;
};
using updatable_cg_params = updatable_coarse_granular_launch_parameters<void>;

#endif
