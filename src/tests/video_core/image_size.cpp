// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <catch2/catch_test_macros.hpp>

#include "video_core/texture_cache/image_size.h"
#include "video_core/texture_cache/util.h"

using namespace VideoCommon;

TEST_CASE("Image sizes reject array byte-count wraparound", "[video_core]") {
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::R16G16B16A16_UNORM;
    info.size = {1920, 1080, 1};
    info.layer_stride = 17'694'720;
    info.resources = {.levels = 1, .layers = 32};
    REQUIRE(CalculateGuestSizeInBytes(info) == 566'231'040);
    REQUIRE(CalculateUnswizzledSizeBytes(info) == 530'841'600);

    // The descriptor observed in Zero Kai must not become a 252 MB backing range.
    info.resources.layers = 257;
    REQUIRE(ImageSize::Multiply(info.layer_stride, 257) == 4'547'543'040);
    REQUIRE_THROWS_AS(CalculateGuestSizeInBytes(info), InvalidImageSize);
    // Its linear upload still fits in u32 and must retain the real size.
    REQUIRE(CalculateUnswizzledSizeBytes(info) == 4'263'321'600);
}

TEST_CASE("Image sizes check intermediate products, shifts and alignment", "[video_core]") {
    constexpr u64 max64 = std::numeric_limits<u64>::max();
    constexpr u64 max32 = std::numeric_limits<u32>::max();
    REQUIRE(ImageSize::Narrow(max32) == max32);
    REQUIRE_THROWS_AS(ImageSize::Narrow(max32 + 1), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::Multiply(max64, 2), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::Add(max64, 1), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::Shift(1, 64), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::Shift(max64, 1), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::AlignUp(max64, 4), InvalidImageSize);
    REQUIRE_THROWS_AS(ImageSize::AlignUp32(max32, 4), InvalidImageSize);
    REQUIRE(ImageSize::DivCeil32(max32, 12) == 357'913'942);
    REQUIRE(ImageSize::AlignUp32(518, 12) == 528);
}

TEST_CASE("Image sizes retain mip and pitch layout calculations", "[video_core]") {
    ImageInfo info;
    info.type = ImageType::e2D;
    info.format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM;
    info.size = {1024, 1024, 1};
    info.block = {0, 4, 0};
    info.resources = {.levels = 3, .layers = 1};
    REQUIRE(CalculateLayerSize(info) == 0x540000);
    const auto offsets = CalculateMipLevelOffsets(info);
    REQUIRE(offsets[0] == 0);
    REQUIRE(offsets[1] == 0x400000);
    REQUIRE(offsets[2] == 0x500000);
    info.type = ImageType::Linear;
    info.pitch = 8192;
    info.size = {1920, 1080, 1};
    info.resources.levels = 1;
    REQUIRE(CalculateGuestSizeInBytes(info) == 8'847'360);
    info.pitch = std::numeric_limits<u32>::max();
    info.size.height = 2;
    REQUIRE_THROWS_AS(CalculateGuestSizeInBytes(info), InvalidImageSize);
}
