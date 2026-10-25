// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/arm/scarlet/module_metadata.h"
#include "core/arm/scarlet/scarlet_code_manager.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace Core::Scarlet {
namespace {

std::uint32_t Read32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t Read64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(Read32(p)) |
           (static_cast<std::uint64_t>(Read32(p + 4)) << 32);
}

bool InImage(std::uint64_t offset, std::uint64_t size, std::size_t image_size) {
    return offset <= image_size && size <= image_size - offset;
}

std::uint64_t Word(const std::uint8_t* p) {
    return sizeof(void*) == 8 ? Read64(p) : Read32(p);
}

// Switch images use common fixed-width DW_EH_PE encodings. Supporting the LEB variants here
// keeps images with compact .eh_frame_hdr tables useful as well.
bool DecodePointer(const std::uint8_t* image, std::size_t limit, std::uint64_t& at,
                   std::uint8_t encoding, std::uint64_t data_base, std::uint64_t& result) {
    const auto field = at;
    const auto format = encoding & 0x0f;
    std::uint64_t value{};
    if (encoding == 0xff || (encoding & 0x80))
        return false;
    switch (format) {
    case 0x00:
        if (!InImage(at, 8, limit))
            return false;
        value = Read64(image + at);
        at += 8;
        break;
    case 0x01:
    case 0x09: {
        unsigned shift = 0;
        std::uint8_t byte{};
        do {
            if (!InImage(at, 1, limit) || shift >= 64)
                return false;
            byte = image[at++];
            value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (format == 0x09 && shift < 64 && (byte & 0x40))
            value |= ~std::uint64_t{} << shift;
        break;
    }
    case 0x02:
    case 0x0a:
        if (!InImage(at, 2, limit))
            return false;
        value = (encoding & 0x08) ? static_cast<std::uint64_t>(
                                        static_cast<std::int16_t>(image[at] | (image[at + 1] << 8)))
                                  : static_cast<std::uint16_t>(image[at] | (image[at + 1] << 8));
        at += 2;
        break;
    case 0x03:
    case 0x0b:
        if (!InImage(at, 4, limit))
            return false;
        value = (encoding & 0x08)
                    ? static_cast<std::uint64_t>(static_cast<std::int32_t>(Read32(image + at)))
                    : Read32(image + at);
        at += 4;
        break;
    case 0x04:
    case 0x0c:
        if (!InImage(at, 8, limit))
            return false;
        value = (encoding & 0x08)
                    ? static_cast<std::uint64_t>(static_cast<std::int64_t>(Read64(image + at)))
                    : Read64(image + at);
        at += 8;
        break;
    default:
        return false;
    }
    switch (encoding & 0x70) {
    case 0x00:
        break;
    case 0x10:
        value += field;
        break;
    case 0x30:
        value += data_base;
        break;
    default:
        return false;
    }
    result = value;
    return true;
}

std::uint64_t RelativeOffset(const std::uint8_t* image, std::uint64_t mod0, unsigned field,
                             std::size_t image_size) {
    const auto displacement = static_cast<std::int32_t>(Read32(image + mod0 + field));
    std::uint64_t target{};
    if (displacement < 0) {
        const auto magnitude = static_cast<std::uint64_t>(-static_cast<std::int64_t>(displacement));
        if (magnitude > mod0)
            return std::numeric_limits<std::uint64_t>::max();
        target = mod0 - magnitude;
    } else {
        target = mod0 + static_cast<std::uint32_t>(displacement);
    }
    return target <= image_size ? target : std::numeric_limits<std::uint64_t>::max();
}

} // namespace

std::vector<std::uint64_t> DiscoverModuleEntryPoints(
    const std::vector<std::uint8_t>& image, std::uint64_t image_address, std::uint64_t text_offset,
    std::uint64_t text_size, std::size_t& unwind_entries, std::size_t& function_symbols,
    std::size_t& relocation_entries) {
    std::vector<std::uint64_t> result;
    unwind_entries = 0;
    function_symbols = 0;
    relocation_entries = 0;
    if (image.size() < 8 || text_offset > image.size() || text_size > image.size() - text_offset)
        return result;
    const auto text_end = text_offset + text_size;
    const auto add = [&](std::uint64_t offset) {
        if (offset >= text_offset && offset < text_end && !(offset & 3))
            result.push_back(image_address + offset);
    };
    const auto mod0 = static_cast<std::uint64_t>(Read32(image.data() + 4));
    if (!InImage(mod0, 24, image.size()) || std::memcmp(image.data() + mod0, "MOD0", 4))
        return result;

    // .eh_frame_hdr provides an authoritative list of function starts, including local functions
    // that cannot be discovered by following calls from the program entry point.
    const auto eh_start = RelativeOffset(image.data(), mod0, 16, image.size());
    const auto eh_end = RelativeOffset(image.data(), mod0, 20, image.size());
    if (eh_start != std::numeric_limits<std::uint64_t>::max() && eh_end >= eh_start &&
        InImage(eh_start, eh_end - eh_start, image.size()) && eh_end - eh_start >= 4 &&
        image[eh_start] == 1) {
        auto at = eh_start + 4;
        std::uint64_t ignored{}, count{};
        const auto table_encoding = image[eh_start + 3];
        if (DecodePointer(image.data(), eh_end, at, image[eh_start + 1], eh_start, ignored) &&
            DecodePointer(image.data(), eh_end, at, image[eh_start + 2], eh_start, count) &&
            count < 4'000'000) {
            for (std::uint64_t i = 0; i < count; ++i) {
                std::uint64_t function{};
                if (!DecodePointer(image.data(), eh_end, at, table_encoding, eh_start, function) ||
                    !DecodePointer(image.data(), eh_end, at, table_encoding, eh_start, ignored))
                    break;
                const auto before = result.size();
                add(function);
                unwind_entries += result.size() != before;
            }
        }
    }

    // Dynamic table and symbol seeds, matching ScarletNX's read_dynamic/read_symbols path.
    constexpr std::uint64_t DT_HASH = 4, DT_STRTAB = 5, DT_SYMTAB = 6, DT_STRSZ = 10,
                            DT_SYMENT = 11, DT_GNU_HASH = 0x6ffffef5;
    std::array<std::uint64_t, 38> dynamic_values{};
    std::array<bool, 38> dynamic_have{};
    const auto dynamic = RelativeOffset(image.data(), mod0, 4, image.size());
    if (dynamic != std::numeric_limits<std::uint64_t>::max()) {
        const auto word = sizeof(void*);
        for (auto at = dynamic; InImage(at, word * 2, image.size()); at += word * 2) {
            const auto tag = Word(image.data() + at);
            const auto value = Word(image.data() + at + word);
            if (!tag)
                break;
            if (tag < dynamic_values.size() && !dynamic_have[tag]) {
                dynamic_values[tag] = value;
                dynamic_have[tag] = true;
            } else if (tag == DT_GNU_HASH && !dynamic_have[37]) {
                dynamic_values[37] = value;
                dynamic_have[37] = true;
            }
        }
    }
    if (dynamic_have[DT_SYMTAB] && dynamic_have[DT_STRTAB]) {
        const auto word = sizeof(void*);
        const auto entry_size = word == 8 ? 24ull : 16ull;
        std::uint64_t count{};
        if (dynamic_have[DT_HASH] && InImage(dynamic_values[DT_HASH], 8, image.size())) {
            count = Read32(image.data() + dynamic_values[DT_HASH] + 4);
        } else if (dynamic_have[37] && InImage(dynamic_values[37], 16, image.size())) {
            const auto gh = dynamic_values[37];
            const auto buckets = Read32(image.data() + gh), first = Read32(image.data() + gh + 4),
                       blooms = Read32(image.data() + gh + 8);
            const auto bucket_at = gh + 16 + static_cast<std::uint64_t>(blooms) * word;
            if (InImage(bucket_at, static_cast<std::uint64_t>(buckets) * 4, image.size())) {
                std::uint32_t last{};
                for (std::uint32_t i = 0; i < buckets; ++i)
                    last = std::max(last, Read32(image.data() + bucket_at + i * 4));
                if (last < first)
                    count = first;
                else {
                    const auto chains = bucket_at + static_cast<std::uint64_t>(buckets) * 4;
                    for (auto index = last;
                         InImage(chains + static_cast<std::uint64_t>(index - first) * 4, 4,
                                 image.size());
                         ++index) {
                        if (Read32(image.data() + chains +
                                   static_cast<std::uint64_t>(index - first) * 4) &
                            1) {
                            count = index + 1;
                            break;
                        }
                    }
                }
            }
        } else if (dynamic_values[DT_STRTAB] > dynamic_values[DT_SYMTAB]) {
            count = (dynamic_values[DT_STRTAB] - dynamic_values[DT_SYMTAB]) / entry_size;
        }
        const auto symtab = dynamic_values[DT_SYMTAB];
        const auto strtab = dynamic_values[DT_STRTAB];
        const auto strsz = dynamic_have[DT_STRSZ] ? dynamic_values[DT_STRSZ] : 0;
        if ((!dynamic_have[DT_SYMENT] || dynamic_values[DT_SYMENT] == entry_size) && count &&
            count < 4'000'000 && InImage(strtab, strsz, image.size()) &&
            InImage(symtab, count * entry_size, image.size())) {
            for (std::uint64_t i = 0; i < count; ++i) {
                const auto* symbol = image.data() + symtab + i * entry_size;
                const auto type = word == 8 ? symbol[4] & 15 : symbol[12] & 15;
                const auto section_index = static_cast<std::uint16_t>(
                    symbol[word == 8 ? 6 : 14] | (symbol[word == 8 ? 7 : 15] << 8));
                const auto value = word == 8 ? Read64(symbol + 8) : Read32(symbol + 4);
                if (type == 2 && section_index) {
                    const auto before = result.size();
                    add(value);
                    function_symbols += result.size() != before;
                }
            }
        }
    }

    // Relative relocations are how vtables, callback arrays and other address-taken functions
    // become seeds in ScarletNX. Decode RELA and REL tables for AArch64 Switch modules.
    constexpr std::uint64_t DT_RELA = 7, DT_RELASZ = 8, DT_RELAENT = 9, DT_REL = 17, DT_RELSZ = 18,
                            DT_RELENT = 19;
    const auto relocate = [&](std::uint64_t addend) {
        const auto before = result.size();
        add(addend);
        relocation_entries += result.size() != before;
    };
    for (const auto& [table_tag, size_tag, ent_tag, words] :
         {std::array<std::uint64_t, 4>{DT_RELA, DT_RELASZ, DT_RELAENT, 3},
          std::array<std::uint64_t, 4>{DT_REL, DT_RELSZ, DT_RELENT, 2}}) {
        if (!dynamic_have[table_tag])
            continue;
        const auto table = dynamic_values[table_tag];
        const auto entry_size = sizeof(void*) * words;
        const auto size = dynamic_have[size_tag] ? dynamic_values[size_tag] : 0;
        if ((dynamic_have[ent_tag] && dynamic_values[ent_tag] != entry_size) || size % entry_size ||
            !InImage(table, size, image.size()))
            continue;
        for (std::uint64_t at = table; at < table + size; at += entry_size) {
            const auto info = Word(image.data() + at + sizeof(void*));
            if ((info & 0xffffffffu) != 1027)
                continue;
            const auto target = Word(image.data() + at);
            if (words == 2 && !InImage(target, sizeof(void*), image.size()))
                continue;
            auto addend = words == 3 ? Word(image.data() + at + 2 * sizeof(void*))
                                     : Word(image.data() + target);
            if (addend >= image_address && addend - image_address < image.size())
                addend -= image_address;
            relocate(addend);
        }
    }
    if (dynamic_have[36]) { // DT_RELR: packed relative relocations
        const auto table = dynamic_values[36];
        const auto size = dynamic_have[35] ? dynamic_values[35] : 0;
        const auto word = sizeof(void*);
        if (size % word == 0 && InImage(table, size, image.size())) {
            std::uint64_t where{};
            for (std::uint64_t at = table; at < table + size; at += word) {
                const auto entry = Word(image.data() + at);
                if (!(entry & 1)) {
                    if (InImage(entry, word, image.size())) {
                        auto addend = Word(image.data() + entry);
                        if (addend >= image_address && addend - image_address < image.size())
                            addend -= image_address;
                        relocate(addend);
                    }
                    where = entry + word;
                } else {
                    for (unsigned bit = 1; bit < word * 8; ++bit)
                        if ((entry >> bit) & 1) {
                            const auto target = where + (bit - 1) * word;
                            if (InImage(target, word, image.size())) {
                                auto addend = Word(image.data() + target);
                                if (addend >= image_address &&
                                    addend - image_address < image.size())
                                    addend -= image_address;
                                relocate(addend);
                            }
                        }
                    where += (word * 8 - 1) * word;
                }
            }
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace Core::Scarlet
