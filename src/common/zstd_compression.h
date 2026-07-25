// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Compresses a source memory region with Zstandard and returns the compressed data in a vector.
 *
 * @param source            The uncompressed source memory region.
 * @param source_size       The size of the uncompressed source memory region.
 * @param compression_level The used compression level. Should be between 1 and 22.
 *
 * @return the compressed data.
 */
[[nodiscard]] std::vector<u8> CompressDataZSTD(const u8* source, std::size_t source_size,
                                               s32 compression_level);

/**
 * Compresses a source memory region with Zstandard with the default compression level and returns
 * the compressed data in a vector.
 *
 * @param source      The uncompressed source memory region.
 * @param source_size The size of the uncompressed source memory region.
 *
 * @return the compressed data.
 */
[[nodiscard]] std::vector<u8> CompressDataZSTDDefault(const u8* source, std::size_t source_size);

/**
 * Decompresses a source memory region with Zstandard and returns the uncompressed data in a vector.
 *
 * @param compressed the compressed source memory region.
 *
 * @return the decompressed data.
 */
[[nodiscard]] std::vector<u8> DecompressDataZSTD(std::span<const u8> compressed);

/**
 * Decompresses a source memory region with Zstandard using an explicit dictionary,
 * and returns the uncompressed data in a vector.
 *
 * Some titles (e.g. The Legend of Zelda: Tears of the Kingdom) compress most of
 * their RomFS assets against a shared dictionary rather than standalone — the
 * frame header carries only a numeric Dictionary_ID, and the actual dictionary
 * bytes must be supplied out of band (typically extracted from a title-specific
 * "ZsDic" archive elsewhere in the RomFS). Passing the wrong dictionary, or none
 * at all, fails with a "Dictionary mismatch" error rather than silently
 * producing garbage output — see max_decompressed_size below for why this
 * matters for validating the right dictionary was used.
 *
 * @param compressed  the compressed source memory region.
 * @param dictionary  the raw dictionary bytes (as extracted from the title's own
 *                    dictionary archive). Must be the exact dictionary the data
 *                    was compressed against, not merely "a" dictionary of the
 *                    right numeric ID — this function does not verify the
 *                    Dictionary_ID itself, only zstd's internal checksum, which
 *                    catches gross mismatches but not necessarily every case.
 * @param max_decompressed_size  hard cap on the output size (bytes), rejecting
 *                    frames that claim to be larger. Unlike
 *                    DecompressDataZSTD(), this has no built-in cap of its own —
 *                    RomFS assets can legitimately be much larger than a
 *                    network packet — so callers should pick a sensible bound
 *                    for the resource type they're extracting.
 *
 * @return the decompressed data, or an empty vector on any failure (frame too
 *         large, corrupt frame, or a dictionary mismatch).
 */
[[nodiscard]] std::vector<u8> DecompressDataZSTDWithDictionary(std::span<const u8> compressed,
                                                                std::span<const u8> dictionary,
                                                                std::size_t max_decompressed_size);

/**
 * Reads the Dictionary_ID a compressed frame says it needs, if any.
 *
 * Some titles (e.g. Tears of the Kingdom) compress most RomFS assets against
 * a shared dictionary rather than standalone. The zstd frame header can carry
 * a small numeric Dictionary_ID identifying which dictionary was used, without
 * embedding the dictionary itself — the actual bytes must come from elsewhere
 * (see DecompressDataZSTDWithDictionary).
 *
 * @param compressed  the compressed source memory region (only the frame
 *                    header needs to be present/valid — the rest of the frame
 *                    doesn't need to be complete for this to work).
 * @return the Dictionary_ID, or 0 if the frame doesn't reference one (a
 *         standalone/non-dictionary frame), or std::nullopt if compressed
 * @return the Dictionary_ID, or 0 if the frame doesn't reference one (a
 *         standalone/non-dictionary frame, or the header couldn't be read
 *         meaningfully — this can't fully distinguish "no dictionary" from
 *         "invalid frame" without ZSTD_isFrame, which isn't available in
 *         every zstd version this project might vendor; an actually invalid
 *         frame will still correctly fail at the decompress step instead),
 *         or std::nullopt if compressed is too short to possibly contain a
 *         complete zstd frame header at all.
 */
[[nodiscard]] std::optional<u32> GetZSTDFrameDictionaryID(std::span<const u8> compressed);

/**
 * Reads the Dictionary_ID embedded in a raw dictionary blob itself (as
 * extracted from a title's own dictionary archive) — the ID a compressed
 * frame's GetZSTDFrameDictionaryID() needs to match for this dictionary to be
 * the right one to decompress it with.
 *
 * @param dictionary  raw dictionary bytes.
 * @return the embedded Dictionary_ID, or 0 if dictionary isn't a "raw content"
 *         dictionary with the standard zstd dictionary magic/header (i.e. it's
 *         being used as a raw content-only dictionary with no formal ID of
 *         its own — treat 0 as "unidentified/none" the same as the frame side).
 */
[[nodiscard]] u32 GetZSTDDictionaryID(std::span<const u8> dictionary);

} // namespace Common::Compression
