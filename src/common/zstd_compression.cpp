// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2025 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <zstd.h>

#include "common/logging.h"
#include "common/zstd_compression.h"

namespace Common::Compression {

std::vector<u8> CompressDataZSTD(const u8* source, std::size_t source_size, s32 compression_level) {
    compression_level = std::clamp(compression_level, 1, ZSTD_maxCLevel());

    const std::size_t max_compressed_size = ZSTD_compressBound(source_size);
    std::vector<u8> compressed(max_compressed_size);

    const std::size_t compressed_size =
        ZSTD_compress(compressed.data(), compressed.size(), source, source_size, compression_level);

    if (ZSTD_isError(compressed_size)) {
        // Compression failed
        return {};
    }

    compressed.resize(compressed_size);

    return compressed;
}

std::vector<u8> CompressDataZSTDDefault(const u8* source, std::size_t source_size) {
    return CompressDataZSTD(source, source_size, ZSTD_CLEVEL_DEFAULT);
}

std::vector<u8> DecompressDataZSTD(std::span<const u8> compressed) {
    if (compressed.empty()) {
        return {};
    }

    const std::size_t decompressed_size =
        ZSTD_getFrameContentSize(compressed.data(), compressed.size());

    // Define a reasonable maximum size for a decompressed network packet.
    // 16 MB is a very generous limit for a single game packet.
    constexpr u64 MAX_REASONABLE_PACKET_SIZE = 16 * 1024 * 1024;

    // ZSTD_CONTENTSIZE_ERROR indicates a corrupted frame or invalid data - reject it
    // ZSTD_CONTENTSIZE_UNKNOWN means the size isn't in the header but decompression can still work
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
        LOG_ERROR(Common, "Received network packet with corrupted or invalid ZSTD frame");
        return {};
    }

    // Reject packets that claim to be larger than reasonable
    if (decompressed_size != ZSTD_CONTENTSIZE_UNKNOWN && decompressed_size > MAX_REASONABLE_PACKET_SIZE) {
        LOG_ERROR(Common, "Received network packet with oversized decompressed_size: {}", decompressed_size);
        return {};
    }

    // When size is unknown, use streaming decompression with a reasonable initial buffer
    if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        // Use streaming decompression for unknown size
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        if (!dctx) {
            LOG_ERROR(Common, "Failed to create ZSTD decompression context");
            return {};
        }

        std::vector<u8> decompressed;
        decompressed.resize(64 * 1024); // Start with 64KB buffer

        ZSTD_inBuffer input = {compressed.data(), compressed.size(), 0};
        ZSTD_outBuffer output = {decompressed.data(), decompressed.size(), 0};

        while (input.pos < input.size) {
            const size_t ret = ZSTD_decompressStream(dctx, &output, &input);
            if (ZSTD_isError(ret)) {
                LOG_ERROR(Common, "ZSTD streaming decompression failed with error: {}", ZSTD_getErrorName(ret));
                ZSTD_freeDCtx(dctx);
                return {};
            }

            // If ret == 0, decompression is complete
            if (ret == 0) {
                break;
            }

            // If output buffer is full but we haven't consumed all input, need more space
            if (output.pos >= output.size && input.pos < input.size) {
                // Double the buffer size, up to maximum
                if (decompressed.size() > MAX_REASONABLE_PACKET_SIZE) {
                    LOG_ERROR(Common, "ZSTD decompressed size exceeds maximum reasonable packet size");
                    ZSTD_freeDCtx(dctx);
                    return {};
                }
                const size_t old_size = decompressed.size();
                decompressed.resize(std::min(old_size * 2, static_cast<size_t>(MAX_REASONABLE_PACKET_SIZE)));
                output.dst = decompressed.data();
                output.size = decompressed.size();
                // Keep output.pos as is - it points to where we continue writing
            }
        }

        // Ensure all data was consumed
        if (input.pos < input.size) {
            LOG_ERROR(Common, "ZSTD streaming decompression: not all input was consumed");
            ZSTD_freeDCtx(dctx);
            return {};
        }

        decompressed.resize(output.pos);
        ZSTD_freeDCtx(dctx);
        return decompressed;
    }

    std::vector<u8> decompressed(decompressed_size);

    const std::size_t uncompressed_result_size = ZSTD_decompress(
        decompressed.data(), decompressed.size(), compressed.data(), compressed.size());

    if (ZSTD_isError(uncompressed_result_size)) { // check the result of decompress
        // Decompression failed
        LOG_ERROR(Common, "ZSTD_decompress failed with error: {}", ZSTD_getErrorName(uncompressed_result_size));
        return {};
    }

    if (decompressed_size != uncompressed_result_size) {
        LOG_ERROR(Common, "ZSTD decompressed size mismatch. Expected {}, got {}", decompressed_size, uncompressed_result_size);
        return {};
    }

    return decompressed;
}

std::vector<u8> DecompressDataZSTDWithDictionary(std::span<const u8> compressed,
                                                  std::span<const u8> dictionary,
                                                  std::size_t max_decompressed_size) {
    if (compressed.empty()) {
        return {};
    }

    const unsigned long long decompressed_size =
        ZSTD_getFrameContentSize(compressed.data(), compressed.size());

    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
        LOG_ERROR(Common, "ZSTD dictionary decompress: corrupted or invalid frame");
        return {};
    }
    if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        LOG_ERROR(Common, "ZSTD dictionary decompress: frame content size unknown — "
                           "streaming-with-dictionary isn't implemented here since every "
                           "known caller (RomFS asset extraction) has a known-size frame");
        return {};
    }
    if (decompressed_size > max_decompressed_size) {
        LOG_ERROR(Common, "ZSTD dictionary decompress: frame claims {} bytes, exceeds cap of {}",
                  decompressed_size, max_decompressed_size);
        return {};
    }

    // Uses ZSTD_decompress_usingDict rather than a prebuilt ZSTD_DDict: simpler
    // lifetime management (no dict object to cache/free across calls), at the
    // cost of re-deriving internal dictionary structures on every call. Fine
    // for a first correctness pass and for one-off extraction; if this becomes
    // a hot path (e.g. scanning many thousands of same-dictionary files),
    // switching to a cached ZSTD_DDict per dictionary is the natural follow-up.
    ZSTD_DCtx* const dctx = ZSTD_createDCtx();
    if (!dctx) {
        LOG_ERROR(Common, "Failed to create ZSTD decompression context");
        return {};
    }

    std::vector<u8> decompressed(decompressed_size);
    const std::size_t result = ZSTD_decompress_usingDict(
        dctx, decompressed.data(), decompressed.size(), compressed.data(), compressed.size(),
        dictionary.data(), dictionary.size());
    ZSTD_freeDCtx(dctx);

    if (ZSTD_isError(result)) {
        LOG_ERROR(Common, "ZSTD_decompress_usingDict failed: {}", ZSTD_getErrorName(result));
        return {};
    }
    if (result != decompressed_size) {
        LOG_ERROR(Common, "ZSTD dictionary decompress: size mismatch, expected {}, got {}",
                  decompressed_size, result);
        return {};
    }

    return decompressed;
}

std::optional<u32> GetZSTDFrameDictionaryID(std::span<const u8> compressed) {
    // A zstd frame header is at most magic(4) + descriptor(1) + window(1) +
    // dict_id(4) + content_size(8) = 18 bytes; well short of that can't be a
    // complete/valid frame header. This avoids depending on
    // ZSTD_FRAMEHEADERSIZE_MIN, which lives behind ZSTD_STATIC_LINKING_ONLY
    // and isn't guaranteed to be defined wherever this compiles.
    constexpr std::size_t kMinPossibleFrameHeaderSize = 18;
    if (compressed.size() < kMinPossibleFrameHeaderSize) {
        return std::nullopt;
    }
    // Deliberately not gated on ZSTD_isFrame() — that function isn't present in
    // every zstd version this project might vendor. ZSTD_getDictID_fromFrame
    // returns 0 both for "no dictionary referenced" and "couldn't read the
    // header", so this can't distinguish those two cases the way a version
    // with ZSTD_isFrame available could — but callers already treat a 0
    // Dictionary_ID as "try plain decompression", and the decompress path
    // itself will correctly fail (via ZSTD_getFrameContentSize returning
    // ZSTD_CONTENTSIZE_ERROR) if the frame is actually invalid, so nothing is
    // silently accepted as valid when it isn't.
    return static_cast<u32>(ZSTD_getDictID_fromFrame(compressed.data(), compressed.size()));
}

u32 GetZSTDDictionaryID(std::span<const u8> dictionary) {
    if (dictionary.empty()) {
        return 0;
    }
    return static_cast<u32>(ZSTD_getDictID_fromDict(dictionary.data(), dictionary.size()));
}

} // namespace Common::Compression
