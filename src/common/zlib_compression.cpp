// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include <zlib.h>

#include "common/zlib_compression.h"

namespace Common::Compression {

namespace {
constexpr size_t kDefaultInitialSize = 4096;
// Sanity bound against a corrupt/hostile stream that would otherwise grow
// the output buffer without limit — same defensive spirit as this
// investigation's other decompressors.
constexpr size_t kMaxOutputSize = 256ull * 1024 * 1024;

std::vector<u8> DecompressDeflate(std::span<const u8> compressed, size_t size_hint,
                                  int window_bits) {
    if (compressed.empty()) {
        return {};
    }

    z_stream strm{};
    if (inflateInit2(&strm, window_bits) != Z_OK) {
        return {};
    }

    strm.next_in = const_cast<Bytef*>(compressed.data());
    strm.avail_in = static_cast<uInt>(compressed.size());

    std::vector<u8> out(std::max<size_t>(size_hint, kDefaultInitialSize));
    size_t total_out = 0;
    int ret = Z_OK;
    while (ret != Z_STREAM_END) {
        if (total_out == out.size()) {
            if (out.size() >= kMaxOutputSize) {
                inflateEnd(&strm);
                return {};
            }
            out.resize(std::min(out.size() * 2, kMaxOutputSize));
        }
        strm.next_out = out.data() + total_out;
        strm.avail_out = static_cast<uInt>(out.size() - total_out);
        ret = inflate(&strm, Z_NO_FLUSH);
        total_out = out.size() - strm.avail_out;
        if (ret != Z_OK && ret != Z_STREAM_END) {
            if (!(ret == Z_BUF_ERROR && strm.avail_out == 0)) {
                inflateEnd(&strm);
                return {};
            }
        }
        if (strm.avail_in == 0 && ret != Z_STREAM_END && strm.avail_out != 0) {
            inflateEnd(&strm);
            return {};
        }
    }
    inflateEnd(&strm);
    out.resize(total_out);
    return out;
}
} // namespace

std::vector<u8> DecompressDataZlib(std::span<const u8> compressed, size_t size_hint) {
    return DecompressDeflate(compressed, size_hint, MAX_WBITS);
}

std::vector<u8> DecompressDataGzip(std::span<const u8> compressed, size_t size_hint) {
    return DecompressDeflate(compressed, size_hint, MAX_WBITS + 16);
}

} // namespace Common::Compression
