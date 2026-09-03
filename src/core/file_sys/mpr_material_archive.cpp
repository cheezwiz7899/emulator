// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <span>

#include "core/file_sys/mpr_material_archive.h"
#include "core/file_sys/vfs/vfs.h"

namespace FileSys {

namespace {

// Bounds-checked little-endian cursor over an in-memory buffer, matching
// the same defensive style as this investigation's other parsers (e.g.
// arc_archive.cpp) — every read either succeeds and advances, or fails
// and leaves the caller free to bail out cleanly.
class Cursor {
public:
    explicit Cursor(std::span<const u8> data_) : data(data_) {}

    [[nodiscard]] bool ReadU8(u8& out) {
        if (pos + 1 > data.size()) return false;
        out = data[pos];
        ++pos;
        return true;
    }
    [[nodiscard]] bool ReadU16(u16& out) {
        if (pos + 2 > data.size()) return false;
        out = static_cast<u16>(data[pos]) | (static_cast<u16>(data[pos + 1]) << 8);
        pos += 2;
        return true;
    }
    [[nodiscard]] bool ReadU32(u32& out) {
        if (pos + 4 > data.size()) return false;
        out = static_cast<u32>(data[pos]) | (static_cast<u32>(data[pos + 1]) << 8) |
              (static_cast<u32>(data[pos + 2]) << 16) | (static_cast<u32>(data[pos + 3]) << 24);
        pos += 4;
        return true;
    }
    [[nodiscard]] bool ReadU64(u64& out) {
        u32 lo, hi;
        if (!ReadU32(lo) || !ReadU32(hi)) return false;
        out = static_cast<u64>(lo) | (static_cast<u64>(hi) << 32);
        return true;
    }
    [[nodiscard]] bool ReadFourCC(std::array<char, 4>& out) {
        if (pos + 4 > data.size()) return false;
        std::memcpy(out.data(), data.data() + pos, 4);
        pos += 4;
        return true;
    }
    [[nodiscard]] bool Skip(u64 n) {
        if (pos + n > data.size()) return false;
        pos += static_cast<size_t>(n);
        return true;
    }
    [[nodiscard]] size_t Position() const { return pos; }
    [[nodiscard]] bool SeekTo(size_t p) {
        if (p > data.size()) return false;
        pos = p;
        return true;
    }
    [[nodiscard]] size_t Size() const { return data.size(); }

private:
    std::span<const u8> data;
    size_t pos = 0;
};

// Sanity bounds against corrupt/hostile counts turning into oversized
// allocations or iteration counts — same defensive spirit as every other
// parser in this investigation.
constexpr u32 kMaxPlausibleCount = 1'000'000;

[[nodiscard]] bool SkipShaderIdentifier(Cursor& c) {
    // mat_technique(1) + perm_unk1(u64) + perm_unk2(u64) = 17 bytes.
    return c.Skip(17);
}

[[nodiscard]] bool SkipShaderToKEntry(Cursor& c) {
    u32 discard32;
    if (!SkipShaderIdentifier(c)) return false;
    if (!c.ReadU32(discard32)) return false; // unk1
    if (!c.ReadU32(discard32)) return false; // unk2
    return true;
}

[[nodiscard]] bool SkipMaterialTocEntry(Cursor& c) {
    u32 name_len{}, shader_entry_count{}, discard32{};
    u16 discard16{};
    if (!c.Skip(16)) return false; // guid
    if (!c.ReadU32(discard32)) return false; // unk1
    if (!c.ReadU32(name_len)) return false;
    if (name_len > kMaxPlausibleCount || !c.Skip(name_len)) return false; // name
    if (!c.ReadU16(discard16)) return false; // unk2
    if (!c.ReadU32(shader_entry_count)) return false;
    if (shader_entry_count > kMaxPlausibleCount) return false;
    for (u32 i = 0; i < shader_entry_count; ++i) {
        if (!SkipShaderToKEntry(c)) return false;
    }
    if (!c.ReadU32(discard32)) return false; // bufferTocAlign
    if (!c.ReadU32(discard32)) return false; // bufferTocSize
    return true;
}

struct ChunkDescriptor {
    std::array<char, 4> id{};
    u64 size = 0;
};

[[nodiscard]] bool ReadChunkDescriptor(Cursor& c, ChunkDescriptor& out) {
    u32 unk{};
    u64 skip{};
    if (!c.ReadFourCC(out.id)) return false;
    if (!c.ReadU64(out.size)) return false;
    if (!c.ReadU32(unk)) return false;
    if (!c.ReadU64(skip)) return false;
    (void)unk;
    return c.Skip(skip);
}

} // namespace

std::vector<MprShaderSource> TryEnumerateMprSnvnShaderSources(const VirtualFile& file) {
    if (!file || file->GetSize() < 32) {
        return {};
    }

    // Upper bound before any read at all: MaterialArchive.arc-sized files are
    // a few MB in every real sample examined, unlike the multi-GB archives
    // this investigation's other parsers had to avoid loading wholesale.
    constexpr u64 kMaxPlausibleFileSize = 512ull * 1024 * 1024;
    if (static_cast<u64>(file->GetSize()) > kMaxPlausibleFileSize) {
        return {}; // Implausibly large for this format — don't attempt a full-file read.
    }
    // Cheap gate before touching real file content: read just the 32-byte
    // FormDescriptor header first and confirm this is genuinely an RFRM/MTRL
    // file before paying for a full read. The original version of this
    // function called ReadAllBytes() unconditionally, before checking any
    // magic bytes at all — for the vast majority of a real ROM's files
    // (everything between 32 bytes and 512MB, i.e. nearly every ordinary
    // texture/audio/model file), that meant a full synchronous read of file
    // contents this function was about to immediately discard once "RFRM"
    // failed to match. Across a few hundred thousand files, that's the
    // difference between this scan finishing and it not visibly progressing
    // at all. Mirrors the same header-first discipline
    // TryEnumerateArcSubFiles/TryEnumerateCpkFiles (arc_archive.cpp,
    // cpk_archive.cpp) already use for exactly this reason.
    const auto header_bytes = file->ReadBytes(32, 0);
    if (header_bytes.size() != 32) {
        return {};
    }
    Cursor header_cursor(header_bytes);
    std::array<char, 4> form_type{}, form_id{};
    u64 form_size{}, form_unk1{};
    u32 reader_version{}, writer_version{};
    if (!header_cursor.ReadFourCC(form_type)) return {};
    if (std::memcmp(form_type.data(), "RFRM", 4) != 0) {
        return {}; // Not an RFRM file — the expected case for every other title.
    }
    if (!header_cursor.ReadU64(form_size)) return {};
    if (!header_cursor.ReadU64(form_unk1)) return {};
    if (!header_cursor.ReadFourCC(form_id)) return {};
    if (std::memcmp(form_id.data(), "MTRL", 4) != 0) {
        return {}; // A different RFRM variant (model, texture, ...) — not this parser's concern.
    }
    if (!header_cursor.ReadU32(reader_version)) return {};
    if (!header_cursor.ReadU32(writer_version)) return {};
    (void)form_unk1;
    (void)writer_version;
    if (reader_version != 22 && reader_version != 10) {
        return {}; // A layout version this parser doesn't understand (raw/unhandled per the real struct definitions).
    }

    // Only now, with a confirmed real MTRL archive, is the full read worth
    // paying for — MaterialArchive.arc-sized files are small enough (a few
    // MB in every real sample examined) to read entirely into memory, unlike
    // the multi-GB archives this investigation's other parsers had to avoid
    // loading wholesale.
    const auto bytes = file->ReadAllBytes();
    Cursor c(bytes);
    if (!c.Skip(32)) return {}; // Already validated as in-bounds by header_bytes.size() == 32 above; this can't actually fail.
    if (32 + form_size > bytes.size()) {
        return {}; // Declared size overruns the actual file — corrupt.
    }
    const size_t archive_end = 32 + static_cast<size_t>(form_size);

    std::vector<MprShaderSource> result;
    while (c.Position() < archive_end) {
        ChunkDescriptor chunk{};
        if (!ReadChunkDescriptor(c, chunk)) break;
        const size_t content_start = c.Position();
        if (chunk.size == 0 || content_start + chunk.size > archive_end) {
            break; // Corrupt chunk size — stop rather than misread the rest of the archive.
        }
        const size_t backend_end = content_start + static_cast<size_t>(chunk.size);
        const bool is_snvn = std::memcmp(chunk.id.data(), "SNVN", 4) == 0;

        u32 flags{};
        bool ok = c.ReadU32(flags);
        if (ok && flags != 0) {
            u32 toc_size{}, buffer_entry_count{}, material_entry_count{};
            ok = ok && c.ReadU32(toc_size);
            ok = ok && c.ReadU32(buffer_entry_count);
            (void)toc_size;
            if (ok && buffer_entry_count > kMaxPlausibleCount) ok = false;
            for (u32 i = 0; ok && i < buffer_entry_count; ++i) {
                u32 discard;
                ok = c.ReadU32(discard) && c.ReadU32(discard); // align, size
            }
            ok = ok && c.ReadU32(material_entry_count);
            if (ok && material_entry_count > kMaxPlausibleCount) ok = false;
            for (u32 i = 0; ok && i < material_entry_count; ++i) {
                ok = SkipMaterialTocEntry(c);
            }
        }

        u32 shader_source_count = 0;
        ok = ok && c.ReadU32(shader_source_count);
        if (ok && shader_source_count > kMaxPlausibleCount) ok = false;
        for (u32 i = 0; ok && i < shader_source_count; ++i) {
            u8 state{};
            u32 str_len{}, data_len{};
            ok = c.ReadU8(state);
            (void)state;
            ok = ok && c.ReadU32(str_len);
            if (ok && str_len > kMaxPlausibleCount) ok = false;
            ok = ok && c.Skip(str_len);
            ok = ok && c.ReadU32(data_len);
            if (!ok) break;
            const size_t data_pos = c.Position();
            if (!c.Skip(data_len)) {
                ok = false;
                break;
            }
            if (is_snvn) {
                result.push_back(MprShaderSource{static_cast<u64>(data_pos), data_len});
            }
        }

        // Whether or not the detailed walk above succeeded, the chunk's own
        // declared size gives an authoritative end position — jump there
        // directly rather than trying to parse the remaining
        // materialImportDesc fields this parser has no use for. This also
        // recovers cleanly from a parse hiccup partway through one backend
        // without losing every backend after it.
        (void)content_start;
        if (!c.SeekTo(backend_end)) {
            break;
        }
    }

    return result;
}

} // namespace FileSys
