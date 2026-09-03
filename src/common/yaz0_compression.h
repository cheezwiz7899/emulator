// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Decompresses a source memory region using Yaz0, Nintendo's LZSS-family
 * compression scheme predating the newer zstd-based RomFS packaging used by
 * titles like Tears of the Kingdom. Still in active use by older-generation
 * or ported titles (e.g. Breath of the Wild, Super Mario Odyssey) for
 * individual SARC-archived resources — most relevantly here, shader archive
 * (.bfsha/.sbfsha) entries.
 *
 * Format: 16-byte header ("Yaz0" magic, big-endian u32 uncompressed size,
 * 8 reserved/alignment bytes historically unused by the decoder), followed
 * by a stream of 8-bit group headers. Each set bit in a group header means
 * "copy one literal byte from the input"; each clear bit means "back-
 * reference": a 2-byte back-reference token encodes a 12-bit distance and a
 * 4-bit length (length 0 in that nibble means an extended length follows in
 * a third byte, biased +0x12; otherwise the nibble is biased +2).
 *
 * @param compressed  the compressed source memory region (must begin with
 *                    the "Yaz0" magic).
 * @return the decompressed data, or an empty vector if compressed doesn't
 *         start with the Yaz0 magic, is truncated/malformed, or decodes to
 *         something inconsistent with its own declared uncompressed size.
 */
[[nodiscard]] std::vector<u8> DecompressDataYaz0(std::span<const u8> compressed);

/**
 * Returns true if data begins with the 4-byte "Yaz0" magic. Cheap enough to
 * call unconditionally as a first check before attempting decompression.
 */
[[nodiscard]] bool IsYaz0(std::span<const u8> data);

} // namespace Common::Compression
