# GC download staging reservation regression tests

This standalone C++20 test extracts the current `StagingBufferRef`, cached-handle
type, request declaration and `StagingBufferPool::RequestGCDownload` body. Only
Vulkan allocation and deferred release are replaced with a fake allocator.
It does not build Android or need a GPU/driver. Python 3 and CMake are required.

```sh
cmake -S tools/gc_download_tests -B build-gc-download-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build-gc-download-tests --config Release
ctest --test-dir build-gc-download-tests -C Release --output-on-failure
```

For MinGW, add `-G "MinGW Makefiles"`. To reproduce the original missing
reservation, configure a separate build with `-DGC_SOURCE_REVISION=279f46ce97`:
the lifetime and concurrent mapped-consumption checks must fail on that baseline.

Coverage includes an active batch, copied handles, a contender while copies are
in flight, a contender after mocked Finish but before mapped-data consumption,
reuse after release, growth, the 32 MiB boundary, host/device OOM, control-block
OOM before reserving storage, and propagation of non-OOM Vulkan errors.
Synchronization uses latches, not timing/sleeps. These tests validate ownership
and request/reuse behavior; they do not validate actual Vulkan completion or
game output. Keep the GC handle alive through real Finish and mapped reads.
