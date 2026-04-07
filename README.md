# FliX: Flipped-Indexing for Scalable GPU Queries and Updates

This repository contains the source code and experimental framework for the paper  
"FliX: Flipped-Indexing for Scalable GPU Queries and Updates."

FliX is a high-performance GPU-resident indexing structure designed to support fast queries and updates.

## Project Structure

The codebase is organized into several main modules, including the FliX implementation and multiple baselines used for performance comparison:
The following 'Index Types' are tested in our work:

- `cg_rtx_index_updates`: implementation of FliX, our coarse-granular index structure optimized for GPU hardware
- `lsm_tree_ashkiani`: LSMu, our optimized variant of the GPU Log-Structured Merge-tree
- `tree_awad`: a high-performance GPU B-Tree implementation
- `hashtable_slab`: the SlabHash dynamic hash table
- `hashtable_warpcore`: the WarpCore hash table library
- `runscripts_experiments/`: Bash scripts to automate benchmark execution and reproduce the figures and graphs presented in the paper

## Benchmarked Data Structures

FliX is benchmarked against the following GPU-resident indexing data structures. If you use these baselines, please cite the original works.

| Index Type | Reference | Link |
| :--- | :--- | :--- |
| FliX | *FliX: Flipped-Indexing for Scalable GPU Queries and Updates* | to be added
| LSMu | Ashkiani et al., *GPU LSM: A Dynamic Dictionary Data Structure for the GPU* | [DOI](https://ieeexplore.ieee.org/document/8425197) |
| GPU-BTree | Awad et al., *Engineering a High-Performance GPU B-Tree* | [DOI](https://dl.acm.org/doi/10.1145/3293883.3295706) |
| Hash_Slab | Ashkiani et al., *A Dynamic Hash Table for the GPU* | [DOI](https://doi.org/10.1109/IPDPS.2018.00052) |
| Hash_Warpcore | Jünger et al., *WarpCore: A Library for Fast Hash Tables on GPUs* | [DOI](https://ieeexplore.ieee.org/document/9406635) |

## Installation Requirements

To build and run FliX, ensure your environment meets the following requirements.

### Hardware

- NVIDIA GPU with support for concurrent kernels and sufficient VRAM for large-scale indexing experiments
- Your system should have at least **32 GB of main memory**.
- All experiments are designed to run on **64-bit Linux**. 


### Software
- Testing on NVIDIA RTX A6000 GPU
- CUDA Toolkit 12.8 or newer
- `gcc` / `g++` version 12.1 or newer
- Ensure your system runs **NVIDIA's 555.42 GPU driver** or newer.


## Usage and Experiments

The project uses a unified entry point in `main.cu`, which invokes the `benchmark_updates` kernel. The index type and experimental parameters are controlled through macros.

To reproduce the experimental results from the paper:

1. Navigate to the `runscripts_experiments` directory.
2. Execute the desired Bash script, such as `./run_throughput_test.sh`.
3. Results will be written to the designated directories for later visualization.

## runscripts and plotting scrips

- plotting scripts and runscripts are being pushed to this repo. 
- This work is in progress ...
- Please check back for updates

## License and Attribution

```cpp
// =============================================================================
// Author:       Rosina Kharal
// Copyright (c) 2025-2026 Justus Henneberg, Rosina Kharal
// SPDX-License-Identifier: GPL-3.0-or-later
// =============================================================================