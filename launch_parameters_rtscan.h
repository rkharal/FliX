#ifndef LAUNCH_PARAMETERS_RTSCAN_H
#define LAUNCH_PARAMETERS_RTSCAN_H

#include "definitions.cuh"


struct rtscan_params {
    OptixTraversableHandle traversable;

    const key32* stored_keys;

    double ray_spacing;

    uint32_t* bitmaps;
    smallsize bitmap_entries;

    const key32* query_lower;
    const key32* query_upper;
    smallsize* result;
    smallsize batch_size;
};

#endif
