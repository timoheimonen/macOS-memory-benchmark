# Implementation Plan: `-cache-size` Parameter

## Overview
Implement a `-cache-size` parameter that allows users to set a custom cache size. When this parameter is set, the benchmark will:
- Skip automatic L1 and L2 cache size detection
- Use the custom cache size for bandwidth and latency benchmarks
- Only perform bandwidth and latency tests for the custom size (not L1 and L2)
- Only print statistics for the custom cache size

## Detailed Implementation Steps

### Step 1: Add Command-Line Argument Parsing ✅ DONE
- Add parsing for `-cache-size` parameter in `main.cpp` argument parsing loop
- Accept size in kilobytes(KB)
- Store the value in a variable (e.g., `custom_cache_size_kilobytes`)
- Add validation to ensure the value is positive and within reasonable bounds (16 KB to 512 MB)
- Minimum is 16 KB (system page size on macOS) - values below this will be rejected

### Step 2: Conditional Cache Size Detection ✅ DONE
- Add a flag to track if custom cache size is being used (e.g., `bool use_custom_cache_size = false`)
- When `-cache-size` is provided, skip calls to `get_l1_cache_size()` and `get_l2_cache_size()`
- Set the flag to true when custom cache size is provided

### Step 3: Modify Cache Buffer Size Calculation ✅ DONE
- Convert custom cache size from KB to bytes: `custom_cache_size_bytes = custom_cache_size_kilobytes * 1024`
- When custom cache size is set, calculate buffer size using 100% of custom size (full custom cache size)
- Apply the same alignment and validation rules:
  - Ensure buffer sizes are multiples of stride (128 bytes)
  - Ensure minimum size (at least 2 pointers worth)
  - Ensure buffer sizes are at least page size aligned (16 KB on macOS)
- If buffer size is less than page size, round up to page size and display an informational message
- Store in a variable like `custom_buffer_size` instead of `l1_buffer_size` and `l2_buffer_size`

### Step 4: Update Memory Allocation Logic ✅ DONE
- When custom cache size is set, only allocate custom cache buffers:
  - Custom cache latency test buffer
  - Custom cache bandwidth test buffers (source and destination)
- Skip L1 and L2 buffer allocations when custom size is set
- Use conditional checks based on the `use_custom_cache_size` flag

### Step 5: Modify Benchmark Execution Loop ✅ DONE
- When custom cache size is set, skip L1 and L2 bandwidth and latency tests
- Only run custom cache bandwidth tests (read, write, copy)
- Only run custom cache latency test
- Use the same warmup and test functions, but with custom buffer sizes

### Step 6: Update `print_cache_info()` Function ✅ DONE
- Modify `print_cache_info()` in `src/utils.cpp` to accept a flag indicating custom mode
- When custom size is set, display custom cache size instead of L1/L2 sizes
- Update the display format to show "Custom Cache Size" instead of "L1 Cache Size" and "L2 Cache Size"

### Step 7: Update `print_results()` Function ✅ DONE
- Modify `print_results()` in `src/utils.cpp` to handle custom cache results
- When custom size is set, display custom cache results instead of L1/L2 sections
- Use generic labels like "Custom Cache" instead of "L1 Cache"/"L2 Cache"
- Update function signature if needed to pass custom cache results

### Step 8: Update `print_statistics()` Function ✅ DONE
- Modify `print_statistics()` in `src/utils.cpp` to only show custom cache statistics when custom size is set
- Skip L1 and L2 statistics sections when in custom mode
- Display custom cache statistics with appropriate labels

### Step 9: Update Help/Usage Text ✅ DONE
- Update `print_usage()` function in `src/utils.cpp` to document the new `-cache-size` parameter
- Specify that the size should be provided in KB (kilobytes)
- Document minimum size requirement (16 KB, system page size)
- Explain the behavior when `-cache-size` is set (skips L1/L2, only tests custom size)
- Add example usage showing how to use the parameter (e.g., `-cache-size 256` for 256 KB)

### Step 10: Update Result Storage Vectors ✅ DONE
- Modify result storage to use custom cache result variables instead of `l1_*` and `l2_*` when custom size is set
- Store results in appropriate vectors for statistics calculation
- Ensure vectors are properly initialized and reserved based on the mode (custom vs. L1/L2)

## Implementation Notes

- **Flag-based approach**: Use a boolean flag (`use_custom_cache_size`) to track whether custom mode is active
- **Code reuse**: Reuse existing buffer calculation and test logic where possible to minimize code duplication
- **Validation**: Ensure custom cache size validation (minimum 16 KB = page size on macOS, maximum 524288 KB = 512 MB)
- **Unit conversion**: Convert KB input to bytes for internal calculations (1 KB = 1024 bytes)
- **Backward compatibility**: When `-cache-size` is not set, behavior should remain completely unchanged
- **Function signatures**: May need to update function signatures in `benchmark.h` and implementations to support custom cache mode
- **Variable naming**: Consider using generic names like `cache_buffer_size` that can work for both L1/L2 and custom modes

## Files to Modify

1. `main.cpp` - Add argument parsing, conditional logic, and benchmark execution
2. `src/utils.cpp` - Update print functions (`print_cache_info`, `print_results`, `print_statistics`, `print_usage`)
3. `src/benchmark.h` - Update function signatures if needed

## Testing Considerations

- Test with `-cache-size` parameter set to various sizes
- Verify that L1/L2 tests are skipped when custom size is set
- Verify that only custom cache statistics are printed
- Test backward compatibility (no `-cache-size` parameter should work as before)
- Test edge cases (minimum valid size 16 KB, very large custom sizes in KB)
- Verify that values below 16 KB are rejected with appropriate error message
- Verify KB to bytes conversion is correct (e.g., 256 KB = 262144 bytes)

