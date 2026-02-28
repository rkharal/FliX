#ifndef LAUNCH_PARAMETERS_FINE_GRANULAR_H
#define LAUNCH_PARAMETERS_FINE_GRANULAR_H

#include "definitions.cuh"


template<typename key_type>
struct fine_granular_launch_parameters {
    OptixTraversableHandle traversable;

    bool long_keys;
    bool has_range_queries;
    bool keys_are_unique;

    // either 32-bit or 64-bit keys, use void if irrelevant
    const key_type* query_lower;
    const key_type* query_upper;

    smallsize* result;
};
using fg_params = fine_granular_launch_parameters<void>;

#endif
