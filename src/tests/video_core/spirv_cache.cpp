// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/spirv_cache.h"

namespace {

class TemporaryCacheFile {
public:
    TemporaryCacheFile() {
        static std::atomic<u64> next_id{};
        const auto base = std::filesystem::temp_directory_path();
        std::error_code ec;
        for (u64 attempt = 0; attempt < 128; ++attempt) {
            directory = base / ("citron_spirv_cache_contract_" +
                                std::to_string(++next_id));
            if (std::filesystem::create_directory(directory, ec)) {
                path = directory / "spirv_cache.bin";
                return;
            }
            ec.clear();
        }
        throw std::runtime_error("could not create private temporary cache directory");
    }

    ~TemporaryCacheFile() {
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    std::filesystem::path path;
    std::filesystem::path directory;
};

VideoCommon::SpirvKey MakeKey(u64 identity) {
    return {
        .unique_hash = identity,
        .stage = Shader::Stage::Fragment,
        .target_key = 0x0102030405060708ULL,
        .cbuf_key = 0x1112131415161718ULL,
        .runtime_key = 0x2122232425262728ULL,
        .texture_key = 0x3132333435363738ULL,
    };
}

Shader::Backend::Bindings MakeBindings() {
    return {
        .unified = 1,
        .uniform_buffer = 2,
        .storage_buffer = 3,
        .texture = 4,
        .image = 5,
        .texture_scaling_index = 6,
        .image_scaling_index = 7,
    };
}

std::vector<char> ReadAllBytes(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file.is_open()) {
        throw std::runtime_error("could not read cache file");
    }
    const auto size = file.tellg();
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    return bytes;
}

TEST_CASE("SPIR-V cache persists explicit binding counters", "[video_core]") {
    TemporaryCacheFile file;
    const auto key = MakeKey(0x123456789abcdef0ULL);
    const auto bindings = MakeBindings();

    VideoCommon::SpirvCache writer;
    writer.Insert(key, {0x07230203, 0x00010600}, bindings);
    writer.Save(file.path);

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    const auto result = reader.Lookup(key, true);
    REQUIRE(result.has_value());
    REQUIRE(*result->spirv == std::vector<u32>{0x07230203, 0x00010600});
    REQUIRE(result->end_binding.unified == bindings.unified);
    REQUIRE(result->end_binding.uniform_buffer == bindings.uniform_buffer);
    REQUIRE(result->end_binding.storage_buffer == bindings.storage_buffer);
    REQUIRE(result->end_binding.texture == bindings.texture);
    REQUIRE(result->end_binding.image == bindings.image);
    REQUIRE(result->end_binding.texture_scaling_index == bindings.texture_scaling_index);
    REQUIRE(result->end_binding.image_scaling_index == bindings.image_scaling_index);
}

TEST_CASE("SPIR-V cache persists scanner provenance", "[video_core]") {
    TemporaryCacheFile file;
    auto key = MakeKey(0x123456789abcdef0ULL);
    key.cbuf_key = 0;

    VideoCommon::SpirvCache writer;
    writer.InsertSpeculative(key.unique_hash, key.stage, key.target_key, key.runtime_key,
                             key.texture_key, {0x07230203}, MakeBindings(), 0, 0, {},
                             VideoCommon::SpirvCacheEntrySource::PrecacheScanner);
    writer.Save(file.path);

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    const auto result = reader.Lookup(key, true);
    REQUIRE(result.has_value());
    REQUIRE(result->is_speculative);
    REQUIRE(result->source == VideoCommon::SpirvCacheEntrySource::PrecacheScanner);
}

TEST_CASE("SPIR-V cache rejects invalid inserts", "[video_core]") {
    VideoCommon::SpirvCache cache;
    const auto bindings = MakeBindings();
    auto zero_identity = MakeKey(0);
    cache.Insert(zero_identity, {0x07230203}, bindings);
    cache.Insert(MakeKey(0x123456789abcdef0ULL), {}, bindings);
    REQUIRE(cache.Size() == 0);
}

TEST_CASE("SPIR-V cache corruption publishes no loaded prefix", "[video_core]") {
    TemporaryCacheFile file;
    const auto bindings = MakeBindings();

    VideoCommon::SpirvCache writer;
    writer.Insert(MakeKey(0x1111111111111111ULL), {0x07230203}, bindings);
    writer.Insert(MakeKey(0x2222222222222222ULL), {0x07230203}, bindings);
    writer.Save(file.path);

    // Each record has one SPIR-V word. Corrupt the source enum in the second
    // record; a decoder must reject the entire file, not retain record one.
    constexpr std::streamoff header_size = 8 + 4 + 4;
    constexpr std::streamoff bytes_before_source = 8 + 1 + 8 * 4 + 4 + 4 + 4 * 7;
    constexpr std::streamoff record_size = bytes_before_source + 1 + 8 * 8;
    std::fstream corrupt{file.path, std::ios::binary | std::ios::in | std::ios::out};
    REQUIRE(corrupt.is_open());
    corrupt.seekp(header_size + record_size + bytes_before_source);
    corrupt.put(static_cast<char>(0xff));
    REQUIRE(corrupt.good());
    corrupt.close();

    VideoCommon::SpirvCache reader;
    reader.Insert(MakeKey(0x3333333333333333ULL), {0x07230203}, bindings);
    reader.Load(file.path);
    REQUIRE(reader.Size() == 0);
    REQUIRE_FALSE(reader.Contains(MakeKey(0x1111111111111111ULL)));
    REQUIRE_FALSE(reader.Contains(MakeKey(0x3333333333333333ULL)));
}

TEST_CASE("SPIR-V cache rejects appended data", "[video_core]") {
    TemporaryCacheFile file;
    const auto key = MakeKey(0x123456789abcdef0ULL);

    VideoCommon::SpirvCache writer;
    writer.Insert(key, {0x07230203}, MakeBindings());
    writer.Save(file.path);
    {
        std::ofstream append{file.path, std::ios::binary | std::ios::app};
        append.put('x');
    }

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    REQUIRE(reader.Size() == 0);
}

TEST_CASE("SPIR-V cache bounds entry count by file size", "[video_core]") {
    TemporaryCacheFile file;
    VideoCommon::SpirvCache writer;
    writer.Insert(MakeKey(0x123456789abcdef0ULL), {0x07230203}, MakeBindings());
    writer.Save(file.path);

    // Header is magic[8], version u32, entry count u32. A corrupt count must
    // be rejected before reserving a huge container.
    std::fstream corrupt{file.path, std::ios::binary | std::ios::in | std::ios::out};
    REQUIRE(corrupt.is_open());
    const u32 impossible_count = std::numeric_limits<u32>::max();
    corrupt.seekp(8 + sizeof(u32));
    corrupt.write(reinterpret_cast<const char*>(&impossible_count), sizeof(impossible_count));
    REQUIRE(corrupt.good());
    corrupt.close();

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    REQUIRE(reader.Size() == 0);
}

TEST_CASE("SPIR-V cache rejects duplicate serialized keys", "[video_core]") {
    TemporaryCacheFile file;
    VideoCommon::SpirvCache writer;
    writer.Insert(MakeKey(0x123456789abcdef0ULL), {0x07230203}, MakeBindings());
    writer.Save(file.path);

    auto bytes = ReadAllBytes(file.path);
    constexpr size_t header_size = 8 + sizeof(u32) + sizeof(u32);
    REQUIRE(bytes.size() > header_size);
    const u32 duplicate_count = 2;
    std::memcpy(bytes.data() + 8 + sizeof(u32), &duplicate_count, sizeof(duplicate_count));
    const std::vector<char> record{bytes.begin() + header_size, bytes.end()};
    bytes.insert(bytes.end(), record.begin(), record.end());
    {
        std::ofstream rewrite{file.path, std::ios::binary | std::ios::trunc};
        rewrite.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    REQUIRE(reader.Size() == 0);
}

TEST_CASE("SPIR-V cache file order ignores insertion order", "[video_core]") {
    TemporaryCacheFile first_file;
    TemporaryCacheFile second_file;
    const auto bindings = MakeBindings();
    const auto low_key = MakeKey(0x1111111111111111ULL);
    const auto high_key = MakeKey(0x2222222222222222ULL);

    VideoCommon::SpirvCache reverse;
    reverse.Insert(high_key, {0x07230203}, bindings);
    reverse.Insert(low_key, {0x07230203}, bindings);
    reverse.Save(first_file.path);

    VideoCommon::SpirvCache forward;
    forward.Insert(low_key, {0x07230203}, bindings);
    forward.Insert(high_key, {0x07230203}, bindings);
    forward.Save(second_file.path);

    REQUIRE(ReadAllBytes(first_file.path) == ReadAllBytes(second_file.path));
}

TEST_CASE("SPIR-V cache recovers the last complete replacement backup", "[video_core]") {
    TemporaryCacheFile file;
    const auto key = MakeKey(0x123456789abcdef0ULL);
    const auto bindings = MakeBindings();

    VideoCommon::SpirvCache writer;
    writer.Insert(key, {0x07230203}, bindings);
    writer.Save(file.path);

    // Save() keeps the old complete file at this name only over the Windows
    // replacement gap. Simulate process termination in that gap.
    const auto backup = file.path.parent_path() / (file.path.filename().string() + ".bak");
    std::filesystem::copy_file(file.path, backup, std::filesystem::copy_options::overwrite_existing);
    std::filesystem::remove(file.path);

    VideoCommon::SpirvCache reader;
    reader.Load(file.path);
    REQUIRE(reader.Contains(key));
}

} // Anonymous namespace
