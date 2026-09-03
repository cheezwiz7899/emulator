// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "common/yaz0_compression.h"

namespace Common::Compression {

namespace {
constexpr u8 kYaz0Magic[4] = {'Y', 'a', 'z', '0'};
constexpr size_t kYaz0HeaderSize = 16;

// Hard cap against a corrupt/hostile header claiming an absurd uncompressed
// size — mirrors the spirit of DecompressDataZSTDWithDictionary's
// max_decompressed_size, but Yaz0's header has no built-in sanity bound of
// its own the way a zstd frame does. RomFS shader archives seen in practice
// top out in the tens of MB (BOTW's largest single shader archive is ~49MB
// decompressed); 256MB leaves generous headroom without accepting an
// obviously-bogus multi-gigabyte claim from a 16-byte header.
constexpr u32 kMaxUncompressedSize = 256u * 1024 * 1024;

u32 ReadBigEndianU32(std::span<const u8> data, size_t offset) {
    return (static_cast<u32>(data[offset]) << 24) | (static_cast<u32>(data[offset + 1]) << 16) |
           (static_cast<u32>(data[offset + 2]) << 8) | static_cast<u32>(data[offset + 3]);
}
} // namespace

bool IsYaz0(std::span<const u8> data) {
    return data.size() >= 4 && std::memcmp(data.data(), kYaz0Magic, 4) == 0;
}

std::vector<u8> DecompressDataYaz0(std::span<const u8> compressed) {
    if (!IsYaz0(compressed) || compressed.size() < kYaz0HeaderSize) {
        return {};
    }

    const u32 uncompressed_size = ReadBigEndianU32(compressed, 4);
    if (uncompressed_size == 0 || uncompressed_size > kMaxUncompressedSize) {
        return {};
    }

    std::vector<u8> out(uncompressed_size);
    size_t src_pos = kYaz0HeaderSize;
    size_t out_pos = 0;
    const size_t src_size = compressed.size();

    while (out_pos < uncompressed_size) {
        if (src_pos >= src_size) {
            return {}; // Truncated: ran out of input mid-stream.
        }
        const u8 group_header = compressed[src_pos++];

        for (int bit = 7; bit >= 0 && out_pos < uncompressed_size; --bit) {
            if (group_header & (1u << bit)) {
                // Literal byte.
                if (src_pos >= src_size) {
                    return {};
                }
                out[out_pos++] = compressed[src_pos++];
                continue;
            }

            // Back-reference: 2-byte token, optionally a 3rd length-extension byte.
            if (src_pos + 2 > src_size) {
                return {};
            }
            const u8 b1 = compressed[src_pos];
            const u8 b2 = compressed[src_pos + 1];
            src_pos += 2;

            const u32 distance = (static_cast<u32>(b1 & 0x0F) << 8) | b2;
            u32 length = b1 >> 4;
            if (length == 0) {
                if (src_pos >= src_size) {
                    return {};
                }
                length = static_cast<u32>(compressed[src_pos++]) + 0x12;
            } else {
                length += 2;
            }

            // copy_src is `distance + 1` bytes behind the current output
            // position — must already be within the written portion of the
            // output buffer, and the copy (which can overlap forward, as
            // with any LZ77-family scheme — that's how runs are encoded) must
            // not run past the declared uncompressed size.
            if (distance + 1 > out_pos) {
                return {}; // Back-reference points before the start of output — corrupt.
            }
            const size_t copy_src = out_pos - (distance + 1);
            if (out_pos + length > uncompressed_size) {
                return {}; // Would overrun the declared size — corrupt/truncated.
            }
            for (u32 i = 0; i < length; ++i) {
                out[out_pos + i] = out[copy_src + i];
            }
            out_pos += length;
        }
    }

    return out;
}

} // namespace Common::Compression
