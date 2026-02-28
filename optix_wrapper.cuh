#ifndef OPTIX_WRAPPER_H
#define OPTIX_WRAPPER_H

#include "cuda_buffer.cuh"


struct optix_wrapper {
    explicit optix_wrapper(bool profiling = false);
    ~optix_wrapper();

    bool profiling;

    CUcontext cuda_context;
    OptixDeviceContext optix_context;
};

#endif
