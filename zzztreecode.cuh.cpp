

template
// File: cg_rtx_index_updates.cuh
// Snippet: add this alias + member inside class cg_rtx_index_updates<...>
private:
    // Use the 4090-like config; tweak flags if you want cache/sort features.
    using static_tree_t = opt_static_tree<
        key_type,
        opt::f::none,            // flags (e.g., opt::f::cache_upper_levels_partial | opt::f::sort_lookups_block)
        /*static_scheduling=*/false,
        /*use_row_layout=*/true,
        /*node_size_log=*/node_size_log,
        /*threads_per_block=*/256,
        /*registers_per_thread=*/4>;
    static_tree_t static_tree_;   // the only tree state you need

public:
    // Replace your existing build_static_tree(...) with this full implementation.
    void build_static_tree(const key_type* keys,
                           size_t size,
                           size_t max_size,
                           size_t available_memory_bytes,
                           double* build_time_ms,
                           size_t* build_bytes)
    {
        smallsize tile_size = TILE_SIZE;
        smallsize warp_size = WARP_SIZE;

        launch_params_buffer.alloc(1); C2EX
        if (build_bytes) *build_bytes += launch_params_buffer.size_in_bytes();

        assert(initial_num_keys_each_node <= node_size);
        partition_count                 = SDIV(size, initial_num_keys_each_node);
        partition_count_with_overflow   = partition_count + 1;
        total_allocation_region_nodes   = additional_nodes_required(max_size, partition_count_with_overflow);

        build_structures_bucket_layer(
            keys,
            size,
            partition_size,
            node_size,
            partition_count,
            partition_count_with_overflow,
            ordered_node_pairs_buffer,
            allocation_buffer,
            maxvalues_buffer,
            copy_buffer,
            node_stride,
            build_time_ms,
            build_bytes,
            initial_num_keys_each_node,
            total_allocation_region_nodes);

        // ---- Build the opt_static_tree over maxvalues (one per partition) ----
        const size_t tree_n = static_cast<size_t>(partition_count_with_overflow);
        //if (tree_n == 0) {
            // Nothing to index; still set up params for downstream code.
            setup_static_tree_build_data<key_type><<<1, 1, 0, 0>>>(
                launch_params_buffer.ptr(),
                /*tree_buffer*/ nullptr,                // not used when empty
                ordered_node_pairs_buffer.ptr(),
                allocation_buffer.ptr(),
                maxvalues_buffer.ptr(),
                node_stride,
                size,
                partition_size,
                node_size,
                copy_buffer,
                partition_count,
                partition_count_with_overflow,
                total_allocation_region_nodes);
            cudaDeviceSynchronize(); C2EX
            return;
        //}

        // Build tree using the per-partition maxima as keys.
        static_tree_.build(
            /*keys=*/maxvalues_buffer.ptr(),
            /*size=*/tree_n,
            /*max_size=*/tree_n,
            /*available_memory_bytes=*/available_memory_bytes,
            /*build_time_ms=*/build_time_ms,
            /*build_bytes=*/build_bytes);

        base_entries_count = tree_n; // if you still rely on this in wrapper

        // Keep your existing param setup (if other kernels need these buffers).
        setup_static_tree_build_data<key_type><<<1, 1, 0, 0>>>(
            launch_params_buffer.ptr(),
            /*tree_buffer: legacy path, not used by opt_static_tree*/ nullptr,
            ordered_node_pairs_buffer.ptr(),
            allocation_buffer.ptr(),
            maxvalues_buffer.ptr(),
            node_stride,
            size,
            partition_size,
            node_size,
            copy_buffer,
            partition_count,
            partition_count_with_overflow,
            total_allocation_region_nodes);

        cudaDeviceSynchronize(); C2EX
    }
