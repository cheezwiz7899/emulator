// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Decompresses CRI Middleware's "CRILAYLA" compression format, used inside
 * CPK archives (One Piece: Pirate Warriors 3's `rom*.cpk` files, and
 * broadly across CRIWARE-licensed titles generally).
 *
 * Format: an 8-byte "CRILAYLA" magic, then a little-endian u32 declaring
 * the compressed body's decompressed size (not including a trailing
 * 0x100-byte uncompressed prefix stored separately), then 4 reserved
 * bytes, then the compressed body, then a verbatim 0x100-byte prefix that
 * becomes the front of the decompressed output. The body itself is a
 * *backwards*-decoded bitstream: read from the end of the buffer toward
 * the start, a 1-bit marker selects either an 8-bit literal byte or a
 * back-reference (13-bit offset, tiered variable-length match length:
 * a 2-bit base, extended by a further 3 bits, then 5 bits, then a chain
 * of 8-bit extensions for very long matches).
 *
 * PROVENANCE AND CONFIDENCE — read before trusting this differently from
 * the rest of this investigation's fixes: the outer CPK/UTF container
 * format (see cpk_archive.h) is a long-public, multiple-independent-
 * source-documented format (cross-checked against an unrelated public
 * gist in addition to the source below) and is trusted at the same level
 * as this investigation's other container parsers. CRILAYLA specifically
 * — the actual compression algorithm, not just where it sits in the file
 * — was understood in full bit-level detail from a single source: the
 * `cricodecs` PyPI package's underlying C++ implementation
 * (`Youjose/PyCriCodecs` on GitHub, fetched directly for understanding
 * only — the repository has no LICENSE file, meaning all rights are
 * reserved by default, so nothing from it is copied here; this is a
 * clean-room implementation of the same documented bit layout). Unlike
 * every other decompressor in this investigation, this one could not be
 * tested against any real compressed sample — no CRILAYLA-compressed data
 * was reachable in the environment this was written in (see
 * docs/precache-scanner/FINDINGS.md section 8a). Implemented with
 * aggressive bounds-checking throughout specifically because of this: a
 * subtle bug in the bit-level logic should produce an early bounds
 * failure (returning an empty vector, causing the caller to skip the
 * file, the same as any other unrecognized format) rather than reading
 * out of bounds or looping unboundedly. Treat this as meaningfully lower-
 * confidence than the rest of this investigation's decompressors until
 * it's been run against real game data.
 *
 * @param data  the source memory region (must begin with the "CRILAYLA" magic).
 * @return the decompressed data, or an empty vector if data doesn't start
 *         with the magic, is truncated/malformed, or decoding hits any
 *         bounds violation.
 */
[[nodiscard]] std::vector<u8> DecompressDataCRILAYLA(std::span<const u8> data);

/**
 * Returns true if data begins with the 8-byte "CRILAYLA" magic.
 */
[[nodiscard]] bool IsCRILAYLA(std::span<const u8> data);

} // namespace Common::Compression
