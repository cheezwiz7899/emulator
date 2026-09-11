# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

add_executable(tests
    common/bit_field.cpp
    common/cityhash.cpp
    common/container_hash.cpp
    common/fibers.cpp
    common/host_memory.cpp
    common/logging.cpp
    common/param_package.cpp
    common/range_map.cpp
    common/ring_buffer.cpp
    common/scratch_buffer.cpp
    common/unique_function.cpp
    common/uint128.cpp
    core/core_timing.cpp
    core/internal_network/network.cpp
    precompiled_headers.h
    video_core/memory_tracker.cpp
    video_core/maxwell_location.cpp
    video_core/precache_compiler_target.cpp
    video_core/shader_program_identity.cpp
    ../video_core/precache_compiler_target.cpp
    ../video_core/precache_cfg_artifact.cpp
    ../video_core/precache_frontend_artifact.cpp
    ../video_core/spirv_cache.cpp
    video_core/spirv_cache.cpp
    input_common/calibration_configuration_job.cpp
)

# Test translation units are not unity-safe.
set_property(TARGET tests PROPERTY UNITY_BUILD OFF)

create_target_directory_groups(tests)

target_link_libraries(tests PRIVATE common core input_common shader_recompiler)
target_link_libraries(tests PRIVATE ${PLATFORM_LIBRARIES} Catch2::Catch2WithMain Threads::Threads)

add_test(NAME tests COMMAND tests)

if (CITRON_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(tests PRIVATE precompiled_headers.h)
endif()
