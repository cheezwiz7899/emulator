// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>

#include "common/crilayla_compression.h"

namespace Common::Compression {

namespace {

constexpr u8 kMagic[8] = {'C', 'R', 'I', 'L', 'A', 'Y', 'L', 'A'};
constexpr size_t kHeaderSize = 0x10;
constexpr size_t kPrefixSize = 0x100;

// Sanity bound against a corrupt/hostile header claiming an absurd body
// size — same defensive spirit as this investigation's other
// decompressors. Real CPK-packed shader-adjacent entries are expected to
// be small; this leaves generous headroom without accepting an obviously
// bogus claim.
constexpr u32 kMaxBodySize = 256u * 1024 * 1024;

// Reads bits from the end of a buffer toward the start — CRILAYLA's body
// is encoded backwards. Every read is bounds-checked against running off
// the front of the buffer; a caller that hits the end mid-decode gets a
// clean "false" rather than an out-of-bounds access.
class ReverseBitReader {
public:
    explicit ReverseBitReader(std::span<const u8> bytes)
        : data(bytes), index(static_cast<s64>(bytes.size()) - 1) {}

    [[nodiscard]] bool Read(u32 bit_count, u32& out) {
        while (bit_buffer_bits < bit_count) {
            if (index < 0) {
                return false; // Ran out of input — truncated/malformed stream.
            }
            bit_buffer = (bit_buffer << 8) | data[static_cast<size_t>(index)];
            --index;
            bit_buffer_bits += 8;
        }
        const u32 mask = bit_count >= 32 ? 0xFFFFFFFFu : ((1u << bit_count) - 1u);
        out = (bit_buffer >> (bit_buffer_bits - bit_count)) & mask;
        bit_buffer_bits -= bit_count;
        return true;
    }

private:
    std::span<const u8> data;
    s64 index;
    u32 bit_buffer = 0;
    u32 bit_buffer_bits = 0;
};

// Tiered variable-length match-length coding: a 2-bit base (values 0-2 are
// final, meaning length = base + 3); if the base is 3, a further 3-bit
// extension is read (values that keep the running total under 10 are
// final); otherwise a 5-bit extension follows, and if that saturates
// (running total hits 41 before the final +3 bias), a chain of 8-bit
// extensions continues for as long as each one reads as 255.
[[nodiscard]] bool ReadMatchLength(ReverseBitReader& reader, u32& out_length) {
    u32 length = 0;
    if (!reader.Read(2, length)) return false;
    if (length < 3) {
        out_length = length + 3;
        return true;
    }
    u32 extra = 0;
    if (!reader.Read(3, extra)) return false;
    length += extra;
    if (length < 10) {
        out_length = length + 3;
        return true;
    }
    if (!reader.Read(5, extra)) return false;
    length += extra;
    if (length == 41) {
        u32 extension = 0;
        do {
            if (!reader.Read(8, extension)) return false;
            length += extension;
            // Guard against an unbounded chain from a malformed/hostile
            // stream — real matches are bounded by the window size, so a
            // chain this long can only be corrupt input, not a real match.
            if (length > kMaxBodySize) return false;
        } while (extension == 255);
    }
    out_length = length + 3;
    return true;
}

} // namespace

bool IsCRILAYLA(std::span<const u8> data) {
    return data.size() >= 8 && std::memcmp(data.data(), kMagic, 8) == 0;
}

std::vector<u8> DecompressDataCRILAYLA(std::span<const u8> data) {
    if (!IsCRILAYLA(data) || data.size() < kHeaderSize + kPrefixSize) {
        return {};
    }

    u32 body_size = 0;
    std::memcpy(&body_size, data.data() + 8, 4);
    if (body_size > kMaxBodySize) {
        return {};
    }

    const size_t body_end = data.size() - kPrefixSize; // Prefix is the last 0x100 bytes of the buffer.
    if (kHeaderSize > body_end) {
        return {}; // Buffer too small to hold both a header and the trailing prefix.
    }
    const std::span<const u8> body = data.subspan(kHeaderSize, body_end - kHeaderSize);
    const std::span<const u8> prefix = data.subspan(body_end, kPrefixSize);

    std::vector<u8> out(static_cast<size_t>(body_size) + kPrefixSize, 0);
    std::memcpy(out.data(), prefix.data(), kPrefixSize);
    if (body_size == 0) {
        return out;
    }

    ReverseBitReader reader(body);
    s64 write_index = static_cast<s64>(out.size()) - 1;

    // Writes one byte at the current (descending) write position; returns
    // true once the position reaches the prefix boundary, signaling the
    // decode loop to stop (the prefix itself is already in place).
    const auto write_byte = [&](u8 byte) -> bool {
        out[static_cast<size_t>(write_index)] = byte;
        if (write_index == static_cast<s64>(kPrefixSize)) {
            --write_index;
            return true;
        }
        --write_index;
        return false;
    };

    while (write_index >= static_cast<s64>(kPrefixSize)) {
        u32 marker = 0;
        if (!reader.Read(1, marker)) return {};

        if (marker == 0) {
            u32 literal = 0;
            if (!reader.Read(8, literal)) return {};
            if (write_byte(static_cast<u8>(literal))) break;
            continue;
        }

        u32 offset = 0;
        if (!reader.Read(13, offset)) return {};
        u32 length = 0;
        if (!ReadMatchLength(reader, length)) return {};

        s64 read_index = write_index + static_cast<s64>(offset) + 3;
        while (length > 0) {
            if (read_index < static_cast<s64>(kPrefixSize) || read_index >= static_cast<s64>(out.size())) {
                return {}; // Back-reference points outside the valid, already-written range — corrupt.
            }
            if (write_byte(out[static_cast<size_t>(read_index)])) break;
            --read_index;
            --length;
        }
    }

    return out;
}

} // namespace Common::Compression
