# SPDX-License-Identifier: GPL-2.0-or-later
cmake_minimum_required(VERSION 3.22)
project(gc_download_tests LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
set(GC_SOURCE_REVISION "" CACHE STRING "Optional baseline Git revision")
set(revision_args)
if(GC_SOURCE_REVISION)
    list(APPEND revision_args --revision "${GC_SOURCE_REVISION}")
endif()
set(generated "${CMAKE_CURRENT_BINARY_DIR}/gc_request.inc")
add_custom_command(OUTPUT "${generated}"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"
        "${generated}" ${revision_args}
    DEPENDS generate.py ../../src/video_core/renderer_vulkan/vk_staging_buffer_pool.h
        ../../src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp)
add_executable(gc_download_tests tests.cpp "${generated}")
target_compile_features(gc_download_tests PRIVATE cxx_std_20)
target_include_directories(gc_download_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gc_download_tests PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(gc_download_tests PRIVATE /W4)
else()
    target_compile_options(gc_download_tests PRIVATE -Wall -Wextra)
endif()
enable_testing()
add_test(NAME gc_download_reservation COMMAND gc_download_tests)
