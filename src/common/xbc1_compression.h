// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Decompresses Monolith Soft's "xbc1" container — used by Xenoblade
 * Chronicles 2 (and, per the studio's engine spanning the series,
 * plausibly other Xenoblade titles) to wrap zlib- or zstd-compressed data
 * referenced from an .arh index file's entries into the paired .ard data
 * file.
 *
 * Format (48-byte header, all fields little-endian):
 *   offset 0x00: u32 magic ("xbc1")
 *   offset 0x04: u32 compression_type (1 = zlib, 3 = zstd)
 *   offset 0x08: u32 uncompressed_size
 *   offset 0x0C: u32 compressed_size
 *   offset 0x10: u32 hash (unused by this decoder)
 *   offset 0x14: char name[28] (unused by this decoder)
 *   offset 0x30: compressed_size bytes of compressed data
 *
 * Derived from the byte layout of a real, GPL-3.0-licensed open-source
 * implementation (PredatorCZ/XenoLib's xbc1.cpp), consulted for
 * understanding only — GPL-3.0 isn't compatible with this codebase's
 * GPL-2.0-or-later license, so nothing from it is copied here; this is a
 * small enough struct to write cleanly from the field layout alone.
 * Verified against real Xenoblade Chronicles 2 data before being trusted —
 * see docs/precache-scanner/FINDINGS.md.
 *
 * @param data  the source memory region (must begin with the "xbc1" magic).
 * @return the decompressed data, or an empty vector if data doesn't start
 *         with the xbc1 magic, is truncated/malformed, names an
 *         unrecognized compression type, or decompression otherwise fails.
 */
[[nodiscard]] std::vector<u8> DecompressDataXBC1(std::span<const u8> data);

/**
 * Returns true if data begins with the 4-byte "xbc1" magic.
 */
[[nodiscard]] bool IsXBC1(std::span<const u8> data);

} // namespace Common::Compression
