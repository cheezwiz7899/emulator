// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include <zlib.h>
#include <zstd.h>

#include "common/xbc1_compression.h"

namespace Common::Compression {

namespace {
constexpr u32 kXbc1Magic = 0x31636278u; // "xbc1" read as LE u32
constexpr size_t kHeaderSize = 0x30;

enum class CompressionType : u32 {
    Zlib = 1,
    Zstd = 3,
};

// Sanity bound against a corrupt/hostile header claiming an absurd
// uncompressed size — same defensive spirit as yaz0_compression.cpp's
// kMaxUncompressedSize. Real XC2 shader-bearing entries seen during the
// investigation this is based on top out under 5MB decompressed.
constexpr u32 kMaxUncompressedSize = 256u * 1024 * 1024;
} // namespace

bool IsXBC1(std::span<const u8> data) {
    if (data.size() < 4) return false;
    u32 magic{};
    std::memcpy(&magic, data.data(), 4);
    return magic == kXbc1Magic;
}

std::vector<u8> DecompressDataXBC1(std::span<const u8> data) {
    if (!IsXBC1(data) || data.size() < kHeaderSize) {
        return {};
    }

    u32 compression_type_raw{}, uncompressed_size{}, compressed_size{};
    std::memcpy(&compression_type_raw, data.data() + 0x04, 4);
    std::memcpy(&uncompressed_size, data.data() + 0x08, 4);
    std::memcpy(&compressed_size, data.data() + 0x0C, 4);

    if (uncompressed_size == 0 || uncompressed_size > kMaxUncompressedSize) {
        return {};
    }
    if (kHeaderSize + compressed_size > data.size()) {
        return {}; // Declared compressed_size overruns what was actually passed in.
    }

    const u8* compressed_data = data.data() + kHeaderSize;
    std::vector<u8> out(uncompressed_size);

    const auto type = static_cast<CompressionType>(compression_type_raw);
    if (type == CompressionType::Zlib) {
        uLongf dest_len = static_cast<uLongf>(out.size());
        const int status = uncompress(reinterpret_cast<Bytef*>(out.data()), &dest_len,
                                       reinterpret_cast<const Bytef*>(compressed_data),
                                       static_cast<uLong>(compressed_size));
        if (status != Z_OK || dest_len != uncompressed_size) {
            return {};
        }
    } else if (type == CompressionType::Zstd) {
        const size_t result = ZSTD_decompress(out.data(), out.size(), compressed_data, compressed_size);
        if (ZSTD_isError(result) || result != uncompressed_size) {
            return {};
        }
    } else {
        return {}; // Unrecognized compression type — fail closed rather than guess.
    }

    return out;
}

} // namespace Common::Compression
