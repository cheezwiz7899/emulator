// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <set>
#include <span>

#include <zstd.h>

#include "core/file_sys/arc_archive.h"
#include "core/file_sys/vfs/vfs.h"

namespace FileSys {

namespace {

constexpr u64 kArcMagic = 0xABCD'EF98'7654'3210ULL;

// A bounds-checked little-endian cursor over an in-memory buffer — every
// read/skip below goes through this, so a truncated or lying table produces
// an empty result (via the caller bailing out on the first failed read)
// rather than an out-of-bounds access. This mirrors the same defensive
// posture as yaz0_compression.cpp and the scanner's own read_u16/u32/u64
// helpers, just centralized here since this parser does many more reads.
class Cursor {
public:
    explicit Cursor(std::span<const u8> data_) : data(data_) {}

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

    [[nodiscard]] bool Skip(size_t n) {
        if (pos + n > data.size()) return false;
        pos += n;
        return true;
    }

    // Skips `count * element_size` bytes, refusing (rather than overflowing
    // or over-reading) if that product doesn't fit in what's left of the
    // buffer. Every count in this table comes straight from the file, so
    // every use of it is treated as untrusted.
    [[nodiscard]] bool SkipCounted(u64 count, size_t element_size) {
        if (element_size != 0 && count > (data.size() - pos) / element_size) return false;
        return Skip(static_cast<size_t>(count) * element_size);
    }

    [[nodiscard]] bool AlignTo(size_t alignment) {
        const size_t aligned = (pos + alignment - 1) / alignment * alignment;
        return Skip(aligned - pos);
    }

    [[nodiscard]] size_t Position() const { return pos; }

private:
    std::span<const u8> data;
    size_t pos = 0;
};

// Struct byte sizes, transcribed field-for-field from
// jam1garner/smash-arc's src/filesystem.rs (see arc_archive.h for
// provenance/caveats). Deliberately not represented as C++ structs mirroring
// the Rust ones — several are bitfields with sub-byte fields that would be
// error-prone to replicate as C++ bitfields with guaranteed-matching layout;
// instead every field this parser actually needs is read individually, in
// file order, with everything else skipped by its known byte width.
constexpr size_t kFileSystemHeaderSize = 88;
constexpr size_t kQuickDirSize = 12;
constexpr size_t kHashToIndexSize = 8;
constexpr size_t kStreamEntrySize = 12;
constexpr size_t kStreamDataSize = 16;
constexpr size_t kFileInfoBucketSize = 8;
constexpr size_t kFilePathSize = 32;
constexpr size_t kFileInfoIndexSize = 8;
constexpr size_t kDirInfoSize = 52;
constexpr size_t kDirectoryOffsetSize = 28; // repr(packed) — no padding
constexpr size_t kFileInfoSize = 16;

} // namespace

std::vector<ArcSubFile> TryEnumerateArcSubFiles(const VirtualFile& file) {
    // Cheap gate before touching the file at all: every title examined
    // during this investigation tops out around 50MB for its single
    // largest legitimate shader/asset file (BOTW's material archive,
    // decompressed); data.arc is 10+ GB. Titles with hundreds of thousands
    // of ordinary small RomFS files would otherwise each pay a small
    // synchronous read-and-check here before the real (parallel) scan even
    // starts — for a single huge file that cost is negligible, but summed
    // across a large ROM's whole file list it isn't. This threshold is
    // deliberately generous (200MB, several times any legitimate file seen)
    // rather than tuned tight, since a false-negative here (skipping a
    // smaller ARC-like file) only means it's scanned the old way — not a
    // crash or data loss.
    constexpr std::size_t kMinPlausibleArcSize = 200ull * 1024 * 1024;
    if (!file || file->GetSize() < kMinPlausibleArcSize) {
        return {};
    }

    // --- Top-level header (0x28 bytes: magic + 3 section offsets + a
    // pointer to the compressed filesystem table). Cheap regardless of how
    // large the overall file is — this is the only read whose cost doesn't
    // scale with the game's actual content size. ---
    const auto header_bytes = file->ReadBytes(0x28, 0);
    if (header_bytes.size() != 0x28) {
        return {};
    }
    Cursor header_cursor(header_bytes);
    u64 magic = 0;
    if (!header_cursor.ReadU64(magic) || magic != kArcMagic) {
        return {}; // Not an ARC file — the normal, expected case for every
                   // other title's RomFS files; not an error.
    }
    u64 file_section_offset = 0, fs_table_offset = 0;
    {
        u64 stream_section_offset_unused = 0, shared_section_offset_unused = 0;
        if (!header_cursor.ReadU64(stream_section_offset_unused)) return {};
        if (!header_cursor.ReadU64(file_section_offset)) return {};
        if (!header_cursor.ReadU64(shared_section_offset_unused)) return {};
        if (!header_cursor.ReadU64(fs_table_offset)) return {};
    }

    if (fs_table_offset + 16 > static_cast<u64>(file->GetSize())) {
        return {};
    }

    // --- CompTableHeader (16 bytes: u32 magic == 0x10, decomp_size,
    // comp_size, section_size — section_size unused here) followed by
    // comp_size bytes of a standalone zstd frame decompressing to the real
    // filesystem table. ---
    const auto comp_table_header = file->ReadBytes(16, fs_table_offset);
    if (comp_table_header.size() != 16) {
        return {};
    }
    Cursor cth_cursor(comp_table_header);
    u32 cth_magic = 0, decomp_size = 0, comp_size = 0;
    if (!cth_cursor.ReadU32(cth_magic) || cth_magic != 0x10) return {};
    if (!cth_cursor.ReadU32(decomp_size)) return {};
    if (!cth_cursor.ReadU32(comp_size)) return {};
    // Sanity bound: the real filesystem table for the full game is a few
    // MB; refuse an unreasonable single allocation from a 32-bit size field
    // (same defensive spirit as yaz0_compression.cpp's kMaxUncompressedSize).
    constexpr u32 kMaxTableSize = 512u * 1024 * 1024;
    if (decomp_size == 0 || decomp_size > kMaxTableSize || comp_size == 0 ||
        comp_size > kMaxTableSize) {
        return {};
    }
    if (fs_table_offset + 16 + comp_size > static_cast<u64>(file->GetSize())) {
        return {};
    }
    const auto compressed_table = file->ReadBytes(comp_size, fs_table_offset + 16);
    if (compressed_table.size() != comp_size) {
        return {};
    }
    // Decompressed directly via libzstd here rather than through
    // Common::Compression::DecompressDataZSTD: that function's fallback
    // path for a frame with no embedded content-size field caps output at
    // 16MB (tuned for network packets, its only other caller) — plausibly
    // too small for this table on a title with hundreds of thousands of
    // files. That ambiguity doesn't apply here anyway: CompTableHeader
    // above already gave an exact, trusted decomp_size, so there's no
    // need to detect or guess it from the frame itself — a single
    // known-exact-size ZSTD_decompress call is both simpler and correct
    // regardless of whether this particular frame happens to embed its own
    // content size.
    std::vector<u8> table(decomp_size);
    const size_t decompress_result =
        ZSTD_decompress(table.data(), table.size(), compressed_table.data(), compressed_table.size());
    if (ZSTD_isError(decompress_result) || decompress_result != decomp_size) {
        return {}; // Decompression failed, or produced an unexpected size.
    }

    Cursor c(table);

    // --- fs_header (88 bytes). Only the count fields this parser actually
    // needs are kept; everything else (table_filesize, the two "always
    // 0x10" unk fields, regional counts, version, etc.) is read into a
    // local and discarded — reading rather than blind-skipping so a
    // truncated buffer is still caught by ReadU32's bounds check at each
    // step, and so the intent of every 4 bytes consumed is visible here
    // rather than folded into one opaque Skip(88). ---
    u32 file_info_path_count = 0, file_info_index_count = 0, folder_count = 0;
    u32 folder_offset_count_1 = 0, hash_folder_count = 0, file_info_count = 0;
    u32 file_info_sub_index_count = 0, file_data_count = 0, folder_offset_count_2 = 0;
    u32 file_data_count_2 = 0, extra_folder = 0, extra_count = 0;
    u32 extra_count_2 = 0, extra_sub_count = 0;
    {
        const size_t start = c.Position();
        u32 discard;
        if (!c.ReadU32(discard)) return {};                      // table_filesize
        if (!c.ReadU32(file_info_path_count)) return {};
        if (!c.ReadU32(file_info_index_count)) return {};
        if (!c.ReadU32(folder_count)) return {};
        if (!c.ReadU32(folder_offset_count_1)) return {};
        if (!c.ReadU32(hash_folder_count)) return {};
        if (!c.ReadU32(file_info_count)) return {};
        if (!c.ReadU32(file_info_sub_index_count)) return {};
        if (!c.ReadU32(file_data_count)) return {};
        if (!c.ReadU32(folder_offset_count_2)) return {};
        if (!c.ReadU32(file_data_count_2)) return {};
        if (!c.ReadU32(discard)) return {};                      // padding
        if (!c.ReadU32(discard)) return {};                      // unk1_10
        if (!c.ReadU32(discard)) return {};                      // unk2_10
        if (!c.Skip(4)) return {};                                // regional_count_1/2 + padding2
        if (!c.ReadU32(discard)) return {};                      // version
        if (!c.ReadU32(extra_folder)) return {};
        if (!c.ReadU32(extra_count)) return {};
        if (!c.Skip(8)) return {};                                // unk[2]
        if (!c.ReadU32(extra_count_2)) return {};
        if (!c.ReadU32(extra_sub_count)) return {};
        if (c.Position() - start != kFileSystemHeaderSize) return {};
    }

    // #[br(align_before = 0x100)] — stream_header starts at the next 0x100
    // boundary after fs_header.
    if (!c.AlignTo(0x100)) return {};

    // --- stream_header (16 bytes: 4 counts, all read since everything that
    // follows needs them to know how much to skip). ---
    u32 quick_dir_count = 0, stream_hash_count = 0, stream_file_index_count = 0,
        stream_offset_entry_count = 0;
    if (!c.ReadU32(quick_dir_count)) return {};
    if (!c.ReadU32(stream_hash_count)) return {};
    if (!c.ReadU32(stream_file_index_count)) return {};
    if (!c.ReadU32(stream_offset_entry_count)) return {};

    // The whole "stream" section (audio/movie, addressed separately by
    // absolute offset with no compression — see arc_archive.h) is skipped
    // wholesale: this parser has no use for anything in it, only for
    // knowing its total byte width so the tables that follow can be found.
    if (!c.SkipCounted(quick_dir_count, kQuickDirSize)) return {};
    if (!c.SkipCounted(stream_hash_count, kHashToIndexSize)) return {}; // stream_hash_to_entries
    if (!c.SkipCounted(stream_hash_count, kStreamEntrySize)) return {}; // stream_entries
    if (!c.SkipCounted(stream_file_index_count, 4)) return {};          // stream_file_indices (u32 each)
    if (!c.SkipCounted(stream_offset_entry_count, kStreamDataSize)) return {}; // stream_datas

    u32 hash_index_group_count = 0, bucket_count = 0;
    if (!c.ReadU32(hash_index_group_count)) return {}; // #[br(temp)]
    if (!c.ReadU32(bucket_count)) return {};            // #[br(temp)]

    if (!c.SkipCounted(bucket_count, kFileInfoBucketSize)) return {};       // file_info_buckets
    if (!c.SkipCounted(hash_index_group_count, kHashToIndexSize)) return {}; // file_hash_to_path_index
    if (!c.SkipCounted(file_info_path_count, kFilePathSize)) return {};     // file_paths
    if (!c.SkipCounted(file_info_index_count, kFileInfoIndexSize)) return {}; // file_info_indices
    if (!c.SkipCounted(folder_count, kHashToIndexSize)) return {};          // dir_hash_to_info_index
    if (!c.SkipCounted(folder_count, kDirInfoSize)) return {};              // dir_infos

    // --- folder_offsets: read for real this time — file_info_to_datas
    // entries reference this table by index to find the base offset their
    // own offset_in_folder is relative to. Only the leading 8-byte `offset`
    // field of each 28-byte DirectoryOffset is kept. ---
    const u64 folder_offset_count =
        static_cast<u64>(folder_offset_count_1) + folder_offset_count_2 + extra_folder;
    std::vector<u64> folder_offsets;
    folder_offsets.reserve(folder_offset_count);
    for (u64 i = 0; i < folder_offset_count; ++i) {
        u64 offset = 0;
        if (!c.ReadU64(offset)) return {};
        if (!c.Skip(kDirectoryOffsetSize - 8)) return {};
        folder_offsets.push_back(offset);
    }

    if (!c.SkipCounted(hash_folder_count, kHashToIndexSize)) return {}; // folder_child_hashes

    const u64 file_info_total =
        static_cast<u64>(file_info_count) + file_data_count_2 + extra_count;
    if (!c.SkipCounted(file_info_total, kFileInfoSize)) return {}; // file_infos — not needed;
                                                                    // this parser walks
                                                                    // file_info_to_datas directly
                                                                    // rather than via file_infos,
                                                                    // since no name/hash lookup is
                                                                    // needed, only every real
                                                                    // sub-file's byte range.

    // --- file_info_to_datas: (folder_offset_index, file_data_index) pairs.
    // Read in full — this is exactly the join table needed to resolve every
    // sub-file's absolute offset. ---
    const u64 file_info_to_data_total =
        static_cast<u64>(file_info_sub_index_count) + file_data_count_2 + extra_count_2;
    std::vector<std::pair<u32, u32>> file_info_to_datas; // (folder_offset_index, file_data_index)
    file_info_to_datas.reserve(file_info_to_data_total);
    for (u64 i = 0; i < file_info_to_data_total; ++i) {
        u32 folder_offset_index = 0, file_data_index = 0;
        if (!c.ReadU32(folder_offset_index)) return {};
        if (!c.ReadU32(file_data_index)) return {};
        if (!c.Skip(4)) return {}; // file_info_index_and_load_type bitfield, unused here
        file_info_to_datas.emplace_back(folder_offset_index, file_data_index);
    }

    // --- file_datas: the actual (offset_in_folder, comp_size, decomp_size,
    // flags) entries. ---
    struct RawFileData {
        u32 offset_in_folder;
        u32 comp_size;
        u32 decomp_size;
        bool is_compressed;
        bool uses_zstd;
    };
    const u64 file_data_total =
        static_cast<u64>(file_data_count) + file_data_count_2 + extra_sub_count;
    std::vector<RawFileData> file_datas;
    file_datas.reserve(file_data_total);
    for (u64 i = 0; i < file_data_total; ++i) {
        u32 offset_in_folder = 0, comp_size = 0, decomp_size = 0, flags = 0;
        if (!c.ReadU32(offset_in_folder)) return {};
        if (!c.ReadU32(comp_size)) return {};
        if (!c.ReadU32(decomp_size)) return {};
        if (!c.ReadU32(flags)) return {};
        file_datas.push_back(RawFileData{offset_in_folder, comp_size, decomp_size,
                                          (flags & 0x1) != 0, (flags & 0x2) != 0});
    }

    // --- Resolve every (folder_offset_index, file_data_index) pair to an
    // absolute (offset, comp_size, decomp_size) triple, exactly matching
    // read_file_data's formula in lookups.rs:
    //   offset = folder_offset + file_section_offset + (offset_in_folder << 2)
    // De-duplicated by absolute offset, since many entries legitimately
    // share one underlying FileData (shared textures/models between
    // characters, palette-swapped costumes, etc.) — without this, a shared
    // file would be independently re-scanned once per thing that
    // references it, which at this file's scale (hundreds of thousands of
    // entries) would multiply the real work several times over for no
    // benefit. ---
    std::vector<ArcSubFile> result;
    std::set<u64> seen_offsets;
    for (const auto& [folder_offset_index, file_data_index] : file_info_to_datas) {
        if (folder_offset_index >= folder_offsets.size()) continue; // corrupt entry — skip, don't abort the whole scan
        if (file_data_index >= file_datas.size()) continue;
        const auto& fd = file_datas[file_data_index];
        const u64 absolute_offset =
            folder_offsets[folder_offset_index] + file_section_offset +
            (static_cast<u64>(fd.offset_in_folder) << 2);
        if (absolute_offset + fd.comp_size > static_cast<u64>(file->GetSize())) {
            continue; // Would read past the end of the file — corrupt entry, skip it.
        }
        if (!seen_offsets.insert(absolute_offset).second) {
            continue; // Already have this one via another (shared) reference.
        }
        result.push_back(ArcSubFile{absolute_offset, fd.comp_size, fd.decomp_size,
                                     fd.is_compressed, fd.uses_zstd});
    }
    return result;
}

} // namespace FileSys
