````markdown
# FliX: Flipped-Indexing for Scalable GPU Queries and Updates

This repository contains the source code and experimental framework for the paper  
*FliX: Flipped-Indexing for Scalable GPU Queries and Updates.*

## Authors

- Rosina Kharal — 
- Trevor Brown — 
- Justus Henneberg — 
- Felix Schuhknecht — 

FliX is a high-performance GPU-resident indexing structure designed to support fast queries and updates.

## Project/Code Structure

The codebase follows a simple benchmark-driven structure:

```text
main.cu
└── calls benchmark_updates
    ├── runs one benchmark configuration
    ├── uses one data structure at a time
    │   └── selected at compile time
    ├── creates an instance of the selected index
    │   ├── FliX       -> impl_cg_rtx_index_updates.cuh
    │   ├── LSMu       -> impl_lsm_tree.cuh
    │   ├── GPU-BTree  -> impl_tree_awad.cuh
    │   ├── SlabHash   -> impl_hashtable_slab.cuh
    │   └── WarpCore   -> impl_hashtable_warpcore.cuh
    └── executes benchmark operations
        ├── index.insert
        ├── index.remove
        ├── index.lookup
        ├── index.successor   (FliX and LSMu only)
        └── index.rebuild     (FliX only)
````

The benchmark entry point is `main.cu`, which calls `benchmark_updates` to execute benchmark experiments. Each experiment uses one data structure at a time, selected at compile time. Within `benchmark_updates`, an instance of the selected index is created based on the chosen implementation. To evaluate all supported data structures, the scripts in `runscripts_experiments/` run the benchmarks repeatedly, one structure at a time.

## Benchmarked Data Structures

The following index types are tested in our experiments. If you use these baselines, please cite the original works.

* `cg_rtx_index_updates`: implementation of FliX, our coarse-granular index structure optimized for GPU hardware
* `lsm_tree_ashkiani`: LSMu, our optimized variant of the GPU Log-Structured Merge-tree
* `tree_awad`: a high-performance GPU B-Tree implementation
* `hashtable_slab`: the SlabHash dynamic hash table
* `hashtable_warpcore`: the WarpCore hash table library

| Index Type    | Reference                                                                   | Link                                                  |
| :------------ | :-------------------------------------------------------------------------- | :---------------------------------------------------- |
| FliX          | *FliX: Flipped-Indexing for Scalable GPU Queries and Updates*               | TBD                                                   |
| LSMu          | Ashkiani et al., *GPU LSM: A Dynamic Dictionary Data Structure for the GPU* | [DOI](https://ieeexplore.ieee.org/document/8425197)   |
| GPU-BTree     | Awad et al., *Engineering a High-Performance GPU B-Tree*                    | [DOI](https://dl.acm.org/doi/10.1145/3293883.3295706) |
| Hash_Slab     | Ashkiani et al., *A Dynamic Hash Table for the GPU*                         | [DOI](https://doi.org/10.1109/IPDPS.2018.00052)       |
| Hash_Warpcore | Jünger et al., *WarpCore: A Library for Fast Hash Tables on GPUs*           | [DOI](https://ieeexplore.ieee.org/document/9406635)   |

## Installation Requirements

To build and run FliX, ensure your environment meets the following requirements.

### Hardware

* NVIDIA GPU with support for concurrent kernels and sufficient VRAM for large-scale indexing experiments
* Tested on an NVIDIA RTX A6000 GPU
* At least 32 GB of main memory
* 64-bit Linux

### Software

* CUDA Toolkit 12.8 or newer
* `gcc` / `g++` version 12.1 or newer
* NVIDIA driver version 555.42 or newer

## Usage and Experiments

The project uses a unified entry point in `main.cu`, which invokes the `benchmark_updates` kernel. The index type and experimental parameters are controlled through macros.

To reproduce the experimental results from the paper:

1. Navigate to the `runscripts_experiments` directory.
2. Run the desired benchmark script.
3. Results will be written to the designated directories for later visualization.

## Run Scripts and Plotting Scripts

* Plotting scripts and run scripts are being added to this repository.
* This section is still in progress.
* Please check back for updates.

## License and Attribution

```cpp
// =============================================================================
// Author:       Rosina Kharal
// Copyright (c) 2025-2026 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================
```