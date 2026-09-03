// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <span>

#include "core/file_sys/cpk_archive.h"
#include "core/file_sys/vfs/vfs.h"

namespace FileSys {

namespace {

// --- Byte-order helpers ------------------------------------------------
// UTF table payloads are big-endian; the outer CPK packet headers wrapping
// them are little-endian. Both conventions appear in this file, so every
// read below is explicit about which one it's using rather than relying
// on a single ambient convention.

[[nodiscard]] bool ReadBE16(std::span<const u8> d, size_t off, u16& out) {
    if (off + 2 > d.size()) return false;
    out = (static_cast<u16>(d[off]) << 8) | d[off + 1];
    return true;
}
[[nodiscard]] bool ReadBE32(std::span<const u8> d, size_t off, u32& out) {
    if (off + 4 > d.size()) return false;
    out = (static_cast<u32>(d[off]) << 24) | (static_cast<u32>(d[off + 1]) << 16) |
          (static_cast<u32>(d[off + 2]) << 8) | static_cast<u32>(d[off + 3]);
    return true;
}
[[nodiscard]] bool ReadBE64(std::span<const u8> d, size_t off, u64& out) {
    u32 hi{}, lo{};
    if (!ReadBE32(d, off, hi) || !ReadBE32(d, off + 4, lo)) return false;
    out = (static_cast<u64>(hi) << 32) | lo;
    return true;
}
[[nodiscard]] bool ReadLE32(std::span<const u8> d, size_t off, u32& out) {
    if (off + 4 > d.size()) return false;
    std::memcpy(&out, d.data() + off, 4);
    return true;
}

// The 64-byte rolling XOR mask CPK uses to lightly obfuscate (not really
// encrypt — this is a well-known, publicly-documented convention, not a
// real security measure) a packet's UTF payload when its enc_flag isn't
// 0xFF. Generated as byte[0] = 0x5F, byte[n] = byte[n-1] * 0x15 (mod 256).
[[nodiscard]] std::array<u8, 64> MakeUtfMask() {
    std::array<u8, 64> mask{};
    u8 value = 0x5F;
    for (auto& b : mask) {
        b = value;
        value = static_cast<u8>(value * 0x15);
    }
    return mask;
}

void UnmaskUtfPayload(std::vector<u8>& payload) {
    static const auto mask = MakeUtfMask();
    for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] ^= mask[i & 63];
    }
}

enum class ColumnFlag : u8 {
    Name = 0x10,
    Default = 0x20,
    Row = 0x40,
};
enum class ColumnType : u8 {
    UInt8 = 0x00,
    SInt8 = 0x01,
    UInt16 = 0x02,
    SInt16 = 0x03,
    UInt32 = 0x04,
    SInt32 = 0x05,
    UInt64 = 0x06,
    SInt64 = 0x07,
    Float = 0x08,
    Double = 0x09,
    String = 0x0A,
    VLData = 0x0B,
    GUID = 0x0C,
};

[[nodiscard]] u32 ColumnTypeSize(ColumnType type) {
    switch (type) {
    case ColumnType::UInt8:
    case ColumnType::SInt8:
        return 1;
    case ColumnType::UInt16:
    case ColumnType::SInt16:
        return 2;
    case ColumnType::UInt32:
    case ColumnType::SInt32:
    case ColumnType::Float:
    case ColumnType::String:
        return 4;
    case ColumnType::UInt64:
    case ColumnType::SInt64:
    case ColumnType::Double:
    case ColumnType::VLData:
        return 8;
    case ColumnType::GUID:
        return 16;
    default:
        return 0;
    }
}

struct Column {
    std::string name;
    ColumnType type{};
    bool is_row = false; // Row flag: value lives per-row in the row data.
                          // If false (constant/Default-only), this parser
                          // doesn't need its value for anything CPK-file-
                          // listing-related, so it's read past and ignored.
};

struct UtfTable {
    std::vector<u8> data; // The full (unmasked) table payload, owned here
                           // since Column/row reads reference it directly
                           // by offset rather than copying strings out
                           // immediately.
    u32 rows_offset = 0;
    u32 strings_offset = 0;
    u32 data_offset = 0;
    u16 row_width = 0;
    u32 row_count = 0;
    std::vector<Column> columns;

    [[nodiscard]] std::optional<std::string> ReadString(u32 string_offset) const {
        const size_t start = static_cast<size_t>(strings_offset) + string_offset;
        if (start >= data.size()) return std::nullopt;
        const size_t end = std::find(data.begin() + static_cast<ptrdiff_t>(start), data.end(), u8{0}) -
                            data.begin();
        return std::string(data.begin() + static_cast<ptrdiff_t>(start), data.begin() + static_cast<ptrdiff_t>(end));
    }

    [[nodiscard]] int FindColumn(std::string_view name) const {
        for (size_t i = 0; i < columns.size(); ++i) {
            if (columns[i].name == name) return static_cast<int>(i);
        }
        return -1;
    }

    // Reads an unsigned integer-typed row value (UInt8/16/32/64 or
    // SInt8/16/32/64 reinterpreted as unsigned, which is fine for the
    // size/offset fields this parser actually uses). Returns nullopt for
    // any other column type, an out-of-range column/row index, or a
    // truncated buffer.
    [[nodiscard]] std::optional<u64> ReadRowUInt(u32 row, int column_index) const {
        if (column_index < 0 || static_cast<size_t>(column_index) >= columns.size()) return std::nullopt;
        const Column& col = columns[column_index];
        if (!col.is_row) return std::nullopt;
        size_t field_offset = 0;
        for (int i = 0; i < column_index; ++i) {
            if (columns[static_cast<size_t>(i)].is_row) {
                field_offset += ColumnTypeSize(columns[static_cast<size_t>(i)].type);
            }
        }
        const size_t row_start = static_cast<size_t>(rows_offset) + static_cast<size_t>(row) * row_width;
        const size_t off = row_start + field_offset;
        switch (col.type) {
        case ColumnType::UInt8:
        case ColumnType::SInt8:
            if (off >= data.size()) return std::nullopt;
            return data[off];
        case ColumnType::UInt16:
        case ColumnType::SInt16: {
            u16 v{};
            if (!ReadBE16(data, off, v)) return std::nullopt;
            return v;
        }
        case ColumnType::UInt32:
        case ColumnType::SInt32: {
            u32 v{};
            if (!ReadBE32(data, off, v)) return std::nullopt;
            return v;
        }
        case ColumnType::UInt64:
        case ColumnType::SInt64: {
            u64 v{};
            if (!ReadBE64(data, off, v)) return std::nullopt;
            return v;
        }
        default:
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<std::string> ReadRowString(u32 row, int column_index) const {
        if (column_index < 0 || static_cast<size_t>(column_index) >= columns.size()) return std::nullopt;
        const Column& col = columns[column_index];
        if (!col.is_row || col.type != ColumnType::String) return std::nullopt;
        size_t field_offset = 0;
        for (int i = 0; i < column_index; ++i) {
            if (columns[static_cast<size_t>(i)].is_row) {
                field_offset += ColumnTypeSize(columns[static_cast<size_t>(i)].type);
            }
        }
        const size_t row_start = static_cast<size_t>(rows_offset) + static_cast<size_t>(row) * row_width;
        u32 string_offset{};
        if (!ReadBE32(data, row_start + field_offset, string_offset)) return std::nullopt;
        return ReadString(string_offset);
    }

    // Reads a VLData-typed row value: an (offset, size) pair, big-endian,
    // relative to `data_offset` (the same relative-addressing convention
    // strings use relative to `strings_offset`). Used for ITOC's nested
    // "DataL"/"DataH" sub-tables — see the ITOC fallback path below.
    [[nodiscard]] std::optional<std::vector<u8>> ReadRowDataRef(u32 row, int column_index) const {
        if (column_index < 0 || static_cast<size_t>(column_index) >= columns.size()) return std::nullopt;
        const Column& col = columns[column_index];
        if (!col.is_row || col.type != ColumnType::VLData) return std::nullopt;
        size_t field_offset = 0;
        for (int i = 0; i < column_index; ++i) {
            if (columns[static_cast<size_t>(i)].is_row) {
                field_offset += ColumnTypeSize(columns[static_cast<size_t>(i)].type);
            }
        }
        const size_t row_start = static_cast<size_t>(rows_offset) + static_cast<size_t>(row) * row_width;
        u32 ref_offset{}, ref_size{};
        if (!ReadBE32(data, row_start + field_offset, ref_offset)) return std::nullopt;
        if (!ReadBE32(data, row_start + field_offset + 4, ref_size)) return std::nullopt;
        const size_t start = static_cast<size_t>(data_offset) + ref_offset;
        if (start + ref_size > data.size()) return std::nullopt;
        return std::vector<u8>(data.begin() + static_cast<ptrdiff_t>(start),
                                data.begin() + static_cast<ptrdiff_t>(start + ref_size));
    }
};

// Parses a UTF table from an already-unmasked payload (magic "@UTF" at
// offset 0). All multi-byte fields inside are big-endian; offsets are
// relative to byte 8 of the table (right after table_size).
[[nodiscard]] std::optional<UtfTable> ParseUtfTable(std::vector<u8> payload) {
    if (payload.size() < 8 || std::memcmp(payload.data(), "@UTF", 4) != 0) {
        return std::nullopt;
    }
    u32 table_size{};
    if (!ReadBE32(payload, 4, table_size)) return std::nullopt;
    // Fields below are relative to offset 8 (right after table_size).
    constexpr size_t kBase = 8;
    u16 version{}, rows_offset16{};
    u32 strings_offset{}, data_offset{}, name_offset{};
    u16 column_count{}, row_width{};
    u32 row_count{};
    if (!ReadBE16(payload, kBase + 0, version)) return std::nullopt;
    if (!ReadBE16(payload, kBase + 2, rows_offset16)) return std::nullopt;
    if (!ReadBE32(payload, kBase + 4, strings_offset)) return std::nullopt;
    if (!ReadBE32(payload, kBase + 8, data_offset)) return std::nullopt;
    if (!ReadBE32(payload, kBase + 12, name_offset)) return std::nullopt;
    if (!ReadBE16(payload, kBase + 16, column_count)) return std::nullopt;
    if (!ReadBE16(payload, kBase + 18, row_width)) return std::nullopt;
    if (!ReadBE32(payload, kBase + 20, row_count)) return std::nullopt;
    (void)version;
    (void)name_offset;
    (void)data_offset;

    // Sanity bound against a corrupt/hostile row_count multiplying out to
    // an oversized allocation.
    constexpr u32 kMaxPlausibleRows = 10'000'000;
    if (row_count > kMaxPlausibleRows) return std::nullopt;

    UtfTable table;
    table.rows_offset = kBase + rows_offset16;
    table.strings_offset = kBase + strings_offset;
    table.data_offset = kBase + data_offset;
    table.row_width = row_width;
    table.row_count = row_count;
    table.data = std::move(payload);

    size_t pos = kBase + 24; // Right after the 24-byte header block read above.
    for (u16 i = 0; i < column_count; ++i) {
        if (pos >= table.data.size()) return std::nullopt;
        const u8 flags = table.data[pos];
        ++pos;
        Column col;
        col.type = static_cast<ColumnType>(flags & 0x0F);
        col.is_row = (flags & static_cast<u8>(ColumnFlag::Row)) != 0;
        const bool has_name = (flags & static_cast<u8>(ColumnFlag::Name)) != 0;
        const bool has_default = (flags & static_cast<u8>(ColumnFlag::Default)) != 0;
        if (has_name) {
            u32 name_str_offset{};
            if (!ReadBE32(table.data, pos, name_str_offset)) return std::nullopt;
            pos += 4;
            auto name = table.ReadString(name_str_offset);
            if (!name) return std::nullopt;
            col.name = std::move(*name);
        }
        if (has_default) {
            // A constant value inline in the schema — this parser doesn't
            // need constant-column values for anything it currently reads
            // (DirName/FileName/FileSize/ExtractSize/FileOffset are all
            // real per-row TOC columns in every real sample this was
            // checked against), so skip past it rather than parse it.
            const u32 size = ColumnTypeSize(col.type);
            if (size == 0 || pos + size > table.data.size()) return std::nullopt;
            pos += size;
        }
        table.columns.push_back(std::move(col));
    }
    return table;
}

// Reads one packet ("CPK ", "TOC ", etc.) at `offset` and returns its
// (unmasked, if applicable) UTF table payload parsed. `declared_size`
// bounds how far the packet is allowed to extend, matching the CPK
// header's own TocSize/ItocSize/etc. field for whichever section this is.
[[nodiscard]] std::optional<UtfTable> LoadPacketUtf(const VirtualFile& file, u64 offset,
                                                       u64 declared_size, const char magic[4]) {
    constexpr size_t kPacketHeaderSize = 0x10;
    if (declared_size < kPacketHeaderSize) return std::nullopt;
    const auto header = file->ReadBytes(kPacketHeaderSize, offset);
    if (header.size() != kPacketHeaderSize) return std::nullopt;
    if (std::memcmp(header.data(), magic, 4) != 0) return std::nullopt;
    u32 enc_flag{}, utf_size{};
    if (!ReadLE32(header, 4, enc_flag)) return std::nullopt;
    if (!ReadLE32(header, 8, utf_size)) return std::nullopt;
    if (static_cast<u64>(kPacketHeaderSize) + utf_size > declared_size) return std::nullopt;

    auto payload = file->ReadBytes(utf_size, offset + kPacketHeaderSize);
    if (payload.size() != utf_size) return std::nullopt;
    if (enc_flag != 0xFF) {
        UnmaskUtfPayload(payload);
    }
    return ParseUtfTable(std::move(payload));
}

} // namespace

std::vector<CpkFileEntry> TryEnumerateCpkFiles(const VirtualFile& file) {
    if (!file || file->GetSize() < 0x800) {
        return {};
    }

    // The root "CPK " packet is fixed at offset 0 with a declared size of
    // 0x800 bytes in every real sample this format was checked against —
    // matches the public documentation and reference implementation this
    // was derived from (see cpk_archive.h).
    auto cpk_header = LoadPacketUtf(file, 0, 0x800, "CPK ");
    if (!cpk_header || cpk_header->row_count == 0) {
        return {}; // Not a CPK file — the expected case for every other title.
    }

    // ContentOffset (where file data actually starts) and Align (files are
    // padded up to this boundary) are used by both the TOC and ITOC paths
    // below.
    u64 content_offset = 0;
    if (const int co_col = cpk_header->FindColumn("ContentOffset"); co_col >= 0) {
        if (const auto v = cpk_header->ReadRowUInt(0, co_col)) {
            content_offset = *v;
        }
    }
    u64 alignment = 0x800;
    if (const int align_col = cpk_header->FindColumn("Align"); align_col >= 0) {
        if (const auto v = cpk_header->ReadRowUInt(0, align_col); v && *v != 0) {
            alignment = *v;
        }
    }
    const auto align_up = [alignment](u64 size) -> u64 {
        return (size + alignment - 1) / alignment * alignment;
    };

    const int toc_offset_col = cpk_header->FindColumn("TocOffset");
    const int toc_size_col = cpk_header->FindColumn("TocSize");
    const auto toc_offset =
        toc_offset_col >= 0 ? cpk_header->ReadRowUInt(0, toc_offset_col) : std::nullopt;
    const auto toc_size = toc_size_col >= 0 ? cpk_header->ReadRowUInt(0, toc_size_col) : std::nullopt;

    if (toc_offset && toc_size) {
        if (auto toc = LoadPacketUtf(file, *toc_offset, *toc_size, "TOC ")) {
            const int dirname_col = toc->FindColumn("DirName");
            const int filename_col = toc->FindColumn("FileName");
            const int filesize_col = toc->FindColumn("FileSize");
            const int extractsize_col = toc->FindColumn("ExtractSize");
            const int fileoffset_col = toc->FindColumn("FileOffset");
            if (filename_col >= 0 && filesize_col >= 0 && fileoffset_col >= 0) {
                std::vector<CpkFileEntry> result;
                result.reserve(toc->row_count);
                for (u32 row = 0; row < toc->row_count; ++row) {
                    auto file_name = toc->ReadRowString(row, filename_col);
                    auto file_size = toc->ReadRowUInt(row, filesize_col);
                    auto file_offset = toc->ReadRowUInt(row, fileoffset_col);
                    if (!file_name || !file_size || !file_offset) {
                        continue; // Corrupt row — skip it, don't abort the whole table.
                    }
                    const u64 absolute_offset = content_offset + *file_offset;
                    if (absolute_offset + *file_size > static_cast<u64>(file->GetSize())) {
                        continue; // Would read past the end of the file — corrupt entry.
                    }
                    CpkFileEntry entry;
                    entry.dir_name = dirname_col >= 0 ? toc->ReadRowString(row, dirname_col).value_or("") : "";
                    entry.file_name = std::move(*file_name);
                    entry.file_offset = absolute_offset;
                    entry.file_size = *file_size;
                    entry.extract_size =
                        extractsize_col >= 0 ? toc->ReadRowUInt(row, extractsize_col).value_or(*file_size)
                                              : *file_size;
                    result.push_back(std::move(entry));
                }
                if (!result.empty()) {
                    return result;
                }
            }
        }
        // TOC present but unusable (missing columns, every row corrupt,
        // etc.) — fall through to ITOC rather than giving up outright.
    }

    // ITOC fallback: some CPKs (and, per its own README's use of "-InFile"-
    // style tests, plausibly the CPKs this fix was written for) list files
    // via an ID-indexed table instead of TOC's direct paths+offsets. ITOC's
    // row 0 holds two counts ("FilesL"/"FilesH") and two VLData columns
    // ("DataL"/"DataH"), each itself a nested "@UTF" table (parseable with
    // the exact same ParseUtfTable used for every other table in this
    // file) whose rows give (ID, FileSize, ExtractSize) — no direct
    // offset. Real offsets are reconstructed by sorting every entry
    // (DataL's rows, then DataH's, up to their respective declared counts)
    // by ID and laying them out sequentially from content_offset, each
    // padded up to `alignment` — exactly mirroring how the reference
    // implementation this was derived from computes it (see
    // docs/precache-scanner/FINDINGS.md).
    const int itoc_offset_col = cpk_header->FindColumn("ItocOffset");
    const int itoc_size_col = cpk_header->FindColumn("ItocSize");
    if (itoc_offset_col < 0 || itoc_size_col < 0) {
        return {}; // Neither TOC nor ITOC usable.
    }
    const auto itoc_offset = cpk_header->ReadRowUInt(0, itoc_offset_col);
    const auto itoc_size = cpk_header->ReadRowUInt(0, itoc_size_col);
    if (!itoc_offset || !itoc_size) {
        return {};
    }
    auto itoc = LoadPacketUtf(file, *itoc_offset, *itoc_size, "ITOC");
    if (!itoc || itoc->row_count == 0) {
        return {};
    }

    const int files_l_col = itoc->FindColumn("FilesL");
    const int files_h_col = itoc->FindColumn("FilesH");
    const int data_l_col = itoc->FindColumn("DataL");
    const int data_h_col = itoc->FindColumn("DataH");
    if (files_l_col < 0 || files_h_col < 0 || data_l_col < 0 || data_h_col < 0) {
        return {};
    }
    const u64 files_l = itoc->ReadRowUInt(0, files_l_col).value_or(0);
    const u64 files_h = itoc->ReadRowUInt(0, files_h_col).value_or(0);
    auto data_l_bytes = itoc->ReadRowDataRef(0, data_l_col);
    auto data_h_bytes = itoc->ReadRowDataRef(0, data_h_col);
    if (!data_l_bytes || !data_h_bytes) {
        return {};
    }
    auto data_l_table = ParseUtfTable(std::move(*data_l_bytes));
    auto data_h_table = ParseUtfTable(std::move(*data_h_bytes));
    if (!data_l_table || !data_h_table) {
        return {};
    }

    struct ItocEntry {
        u64 id;
        u64 file_size;
        u64 extract_size;
    };
    std::vector<ItocEntry> itoc_entries;

    const auto collect = [&](const UtfTable& table, u64 row_limit) {
        const int id_col = table.FindColumn("ID");
        const int size_col = table.FindColumn("FileSize");
        const int extract_col = table.FindColumn("ExtractSize");
        if (id_col < 0 || size_col < 0) return;
        const u64 limit = std::min<u64>(row_limit, table.row_count);
        for (u64 row = 0; row < limit; ++row) {
            auto id = table.ReadRowUInt(static_cast<u32>(row), id_col);
            auto size = table.ReadRowUInt(static_cast<u32>(row), size_col);
            if (!id || !size) continue; // Corrupt row — skip it, don't abort the whole table.
            const u64 extract = extract_col >= 0
                                     ? table.ReadRowUInt(static_cast<u32>(row), extract_col).value_or(*size)
                                     : *size;
            itoc_entries.push_back(ItocEntry{*id, *size, extract});
        }
    };
    collect(*data_l_table, files_l);
    collect(*data_h_table, files_h);
    if (itoc_entries.empty()) {
        return {};
    }

    // Stable sort by ID — matches the reference implementation's ordering
    // exactly (it additionally radix-sorts for large file counts as a pure
    // performance optimization; functionally equivalent to this for
    // correctness purposes, so not replicated here).
    std::stable_sort(itoc_entries.begin(), itoc_entries.end(),
                      [](const ItocEntry& a, const ItocEntry& b) { return a.id < b.id; });

    std::vector<CpkFileEntry> result;
    result.reserve(itoc_entries.size());
    u64 running_offset = content_offset;
    for (const auto& e : itoc_entries) {
        const u64 absolute_offset = running_offset;
        running_offset += align_up(e.file_size);
        if (absolute_offset + e.file_size > static_cast<u64>(file->GetSize())) {
            continue; // Would read past the end of the file — corrupt entry, but keep computing
                       // subsequent offsets from the (still-valid) running total, since ITOC
                       // entries are laid out sequentially and one bad size shouldn't shift
                       // every entry after it out of alignment with the real file.
        }
        CpkFileEntry entry;
        entry.dir_name = "";
        entry.file_name = "itoc_id_" + std::to_string(e.id);
        entry.file_offset = absolute_offset;
        entry.file_size = e.file_size;
        entry.extract_size = e.extract_size;
        result.push_back(std::move(entry));
    }
    return result;
}

} // namespace FileSys
