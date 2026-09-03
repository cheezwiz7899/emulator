// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::Compression {

/**
 * Decompresses a standard zlib-wrapped deflate stream (the "78 xx" header
 * format — RFC 1950) without needing to know the true decompressed size in
 * advance. Grows its output buffer as needed rather than trusting a
 * caller-supplied or format-declared size hint, which — unlike the size
 * fields accompanying zstd frames or this codebase's other compressed
 * formats — was found during this investigation to sometimes be simply
 * wrong: Metroid Prime Remastered's `SNVN`-backend shader sources declare
 * a `decompressed_size` field that doesn't match the stream's actual
 * output length (confirmed against real data — a fixed-size
 * `uncompress()` call using that field fails with a buffer-too-small
 * error, while decompressing without a size assumption succeeds and
 * produces the real, larger output). See
 * docs/precache-scanner/FINDINGS.md section 3.
 *
 * @param compressed    the zlib-wrapped compressed source data.
 * @param size_hint     an initial output buffer size to allocate before any
 *                       growth — purely a performance hint (avoids a few
 *                       reallocations when a roughly-correct size is known),
 *                       never trusted as authoritative. Pass 0 to use a
 *                       small default.
 * @return the decompressed data, or an empty vector if `compressed` isn't
 *         a valid zlib stream or decompression otherwise fails.
 */
[[nodiscard]] std::vector<u8> DecompressDataZlib(std::span<const u8> compressed, size_t size_hint = 0);

} // namespace Common::Compression
