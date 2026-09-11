// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/cityhash.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/memory_manager.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_program_identity.h"
#include "video_core/texture_cache/format_lookup_table.h"
#include "video_core/textures/texture.h"

namespace VideoCommon {

namespace {

bool IsValidShaderEntryInstruction(u32 initial_offset, u32 start_address, u32 code_lowest,
                                   u32 code_highest, std::span<const u64> code) noexcept {
    u32 entry_address = start_address + initial_offset;
    // Maxwell stores one scheduling control word at the start of each 32-byte instruction group.
    // Match Location::Align() so validation checks the first instruction, not its scheduler word.
    if (entry_address % 32 == 0) {
        entry_address += sizeof(u64);
    }
    if (entry_address < code_lowest || entry_address > code_highest ||
        (entry_address - code_lowest) % sizeof(u64) != 0) {
        return false;
    }
    const size_t index = (entry_address - code_lowest) / sizeof(u64);
    return index < code.size() && Shader::Maxwell::TryDecode(code[index]).has_value();
}

} // Anonymous namespace

constexpr std::array<char, 8> MAGIC_NUMBER{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

constexpr size_t INST_SIZE = sizeof(u64);
// Per-environment limits for transferable-cache input. These are intentionally
// much larger than real shader captures, but prevent a corrupt count from
// turning a cache miss into an unbounded allocation or loop during startup.
constexpr u64 MAX_SERIALIZED_SHADER_BYTES = 64ULL * 1024 * 1024;
constexpr u64 MAX_SERIALIZED_ENV_ENTRIES = 1ULL * 1024 * 1024;
constexpr u64 MAX_SERIALIZED_PIPELINE_RECORD_BYTES = 512ULL * 1024 * 1024;

bool IsValidTextureType(Shader::TextureType type) noexcept {
    return static_cast<u32>(type) < Shader::NUM_TEXTURE_TYPES;
}

bool IsValidTexturePixelFormat(Shader::TexturePixelFormat format) noexcept {
    return static_cast<u32>(format) <= static_cast<u32>(Shader::TexturePixelFormat::D32_FLOAT_S8_UINT);
}

bool IsValidReplaceConstant(Shader::ReplaceConstant value) noexcept {
    return static_cast<u32>(value) <= static_cast<u32>(Shader::ReplaceConstant::DrawID);
}

static Shader::TextureType ConvertTextureType(const Tegra::Texture::TICEntry& entry) {
    switch (entry.texture_type) {
    case Tegra::Texture::TextureType::Texture1D:
        return Shader::TextureType::Color1D;
    case Tegra::Texture::TextureType::Texture2D:
    case Tegra::Texture::TextureType::Texture2DNoMipmap:
        return entry.normalized_coords ? Shader::TextureType::Color2D
                                       : Shader::TextureType::Color2DRect;
    case Tegra::Texture::TextureType::Texture3D:
        return Shader::TextureType::Color3D;
    case Tegra::Texture::TextureType::TextureCubemap:
        return Shader::TextureType::ColorCube;
    case Tegra::Texture::TextureType::Texture1DArray:
        return Shader::TextureType::ColorArray1D;
    case Tegra::Texture::TextureType::Texture2DArray:
        return Shader::TextureType::ColorArray2D;
    case Tegra::Texture::TextureType::Texture1DBuffer:
        return Shader::TextureType::Buffer;
    case Tegra::Texture::TextureType::TextureCubeArray:
        return Shader::TextureType::ColorArrayCube;
    default:
        UNIMPLEMENTED();
        return Shader::TextureType::Color2D;
    }
}

static Shader::TexturePixelFormat ConvertTexturePixelFormat(const Tegra::Texture::TICEntry& entry) {
    return static_cast<Shader::TexturePixelFormat>(
        PixelFormatFromTextureInfo(entry.format, entry.r_type, entry.g_type, entry.b_type,
                                   entry.a_type, entry.srgb_conversion));
}

static std::string_view StageToPrefix(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::VertexB:
        return "VB";
    case Shader::Stage::TessellationControl:
        return "TC";
    case Shader::Stage::TessellationEval:
        return "TE";
    case Shader::Stage::Geometry:
        return "GS";
    case Shader::Stage::Fragment:
        return "FS";
    case Shader::Stage::Compute:
        return "CS";
    case Shader::Stage::VertexA:
        return "VA";
    default:
        return "UK";
    }
}

static void DumpImpl(u64 pipeline_hash, u64 shader_hash, std::span<const u64> code,
                     [[maybe_unused]] u32 read_highest, [[maybe_unused]] u32 read_lowest,
                     u32 initial_offset, Shader::Stage stage) {
    const auto shader_dir{Common::FS::GetCitronPath(Common::FS::CitronPath::DumpDir)};
    const auto base_dir{shader_dir / "shaders"};
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create shader dump directories");
        return;
    }
    const auto prefix = StageToPrefix(stage);
    const auto name{base_dir /
                    fmt::format("{:016x}_{}_{:016x}.ash", pipeline_hash, prefix, shader_hash)};
    std::fstream shader_file(name, std::ios::out | std::ios::binary);
    ASSERT(initial_offset % sizeof(u64) == 0);
    const size_t jump_index = initial_offset / sizeof(u64);
    const size_t code_size = code.size_bytes() - initial_offset;
    shader_file.write(reinterpret_cast<const char*>(&code[jump_index]), code_size);

    // + 1 instruction, due to the fact that we skip the final self branch instruction in the code,
    // but we need to consider it for padding, otherwise nvdisasm rages.
    const size_t padding_needed = (32 - ((code_size + INST_SIZE) % 32)) % 32;
    for (size_t i = 0; i < INST_SIZE + padding_needed; i++) {
        shader_file.put(0);
    }
}

GenericEnvironment::GenericEnvironment(Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_)
    : gpu_memory{&gpu_memory_}, program_base{program_base_} {
    start_address = start_address_;
}

GenericEnvironment::~GenericEnvironment() = default;

u32 GenericEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

u32 GenericEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 GenericEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

std::array<u32, 3> GenericEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

u64 GenericEnvironment::ReadInstruction(u32 address) {
    read_lowest = std::min(read_lowest, address);
    read_highest = std::max(read_highest, address);

    if (address >= cached_lowest && address < cached_highest) {
        return code[(address - cached_lowest) / INST_SIZE];
    }
    has_unbound_instructions = true;
    return gpu_memory->Read<u64>(program_base + address);
}

std::optional<u64> GenericEnvironment::Analyze() {
    const std::optional<u64> size{TryFindSize()};
    if (!size) {
        return std::nullopt;
    }
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(*size);
    return ComputeMaxwellProgramIdentity(code, static_cast<size_t>(*size));
}

void GenericEnvironment::SetCachedSize(size_t size_bytes) {
    cached_lowest = start_address;
    cached_highest = start_address + static_cast<u32>(size_bytes);
    code.resize(CachedSizeWords());
    gpu_memory->ReadBlock(program_base + cached_lowest, code.data(), code.size() * sizeof(u64));
}

size_t GenericEnvironment::CachedSizeWords() const noexcept {
    return CachedSizeBytes() / INST_SIZE;
}

size_t GenericEnvironment::CachedSizeBytes() const noexcept {
    return static_cast<size_t>(cached_highest) - cached_lowest + INST_SIZE;
}

size_t GenericEnvironment::ReadSizeBytes() const noexcept {
    return read_highest - read_lowest + INST_SIZE;
}

bool GenericEnvironment::CanBeSerialized() const noexcept {
    return !has_unbound_instructions &&
           IsValidShaderEntryInstruction(initial_offset, start_address, cached_lowest,
                                         cached_highest, code);
}

u64 GenericEnvironment::CalculateHash() const {
    const size_t size{ReadSizeBytes()};
    const auto data{std::make_unique<char[]>(size)};
    gpu_memory->ReadBlock(program_base + read_lowest, data.get(), size);
    return Common::CityHash64(data.get(), size);
}

namespace {
// Texture-type specialization feasibility instrumentation.
// see handoff_04_specialization_constants_investigation.md item 2 and the doc comments on
// RecordResolvedTextureType()/RecordResolvedTexturePixelFormat() in environment.h.
//
// TextureType and TexturePixelFormat: real, confirmed against 5 real sessions across 5
// different games (TotK, BOTW, Metroid Prime Remastered, Super Mario Odyssey, SSBU) — see
// handoff_09/handoff_10. Result: TextureType varied in 16 of 8671 observed (shader, slot)
// pairs (0.185%); TexturePixelFormat varied in 0 of 8671, a dead end on real content so far.
// IsTexturePixelFormatInteger below is the newest addition, added specifically to keep
// looking for other axes now that the first two have real numbers behind them rather than
// concluding no other axis could exist — not yet tested against any real session itself.
//
// Keyed by (unique_hash, MakeCbufKey(cbuf_index, cbuf_offset)) — unique_hash is memoized
// per-GenericEnvironment-instance (texture_slot_diag_hash_cache, shader_environment.h) rather
// than recomputed per call, since CalculateHash() re-reads and re-hashes the whole shader's
// GPU-memory bytes every time and this can fire once per texture instruction in a shader.
//
// Guarded by one mutex rather than anything lock-free: this only fires on a shader
// TRANSLATION (i.e. a cache MISS — see LogTextureSlotVarianceReportThrottled()'s doc comment
// for why a cache HIT can never introduce a new distinct value and is therefore safe to not
// separately observe), which is already infrequent relative to draw calls, so a mutex is
// simpler and safer here than optimizing for contention that most likely does not exist.
//
// Three independent maps, not one combined struct: see RecordResolvedTextureType/
// RecordResolvedTexturePixelFormat's shared doc comment in environment.h for why the reads
// aren't reliably paired at a single call site — IsTexturePixelFormatInteger is called from
// yet another, mostly-disjoint set of sites (texture_pass.cpp's SNORM-workaround path plus
// wherever an integer-sampled texture instruction appears), so the same reasoning applies a
// third time, not just by analogy.
std::mutex g_texture_slot_variance_mutex;
std::unordered_map<u64, std::unordered_map<u64, std::unordered_set<Shader::TextureType>>>
    g_texture_types_seen;
std::unordered_map<u64, std::unordered_map<u64, std::unordered_set<Shader::TexturePixelFormat>>>
    g_pixel_formats_seen;
std::unordered_map<u64, std::unordered_map<u64, std::unordered_set<bool>>> g_is_integer_seen;

// Adaptive slot learning: coordinates recorded as candidates this
// session, guarded by g_texture_slot_variance_mutex above since they're only ever touched
// from inside RecordResolvedTextureType (write) and TakePhase4PrototypeCandidates (read),
// both already under that lock or taking it themselves. A std::vector, not a set: real counts
// here are tiny (order of 1-10 per session, matching the variance-event rate this whole
// mechanism has always seen) and insertion order doesn't matter, so the O(n) dedup check in
// the write path costs nothing measurable and avoids pulling in a second hasher/set type for
// a container this small.
std::vector<Shader::Phase4PrototypeSlot> g_phase4_prototype_candidates;

// Matches SpirvCache::SaveThrottled's throttling intent (spirv_cache.cpp) — bound log volume
// over a long session instead of printing on every call. A flat call-count window (not
// time-based) is enough since call frequency here already tracks translation frequency,
// itself already infrequent relative to draws.
constexpr size_t kLogEveryNCalls = 200;
std::atomic<size_t> g_report_call_count{0};
} // namespace

std::vector<Shader::Phase4PrototypeSlot> TakePhase4PrototypeCandidates() {
    std::scoped_lock lock{g_texture_slot_variance_mutex};
    return g_phase4_prototype_candidates;
}

void GenericEnvironment::RecordResolvedTextureType(const Shader::TextureSlot& slot, u32 handle,
                                                    Shader::TextureType type) {
    logical_texture_slots[slot].type = type;
    logical_texture_handles.insert(handle);
    const u32 cbuf_index{slot.cbuf_index};
    const u32 cbuf_offset{slot.cbuf_offset};
    if (!texture_slot_diag_hash_cache) {
        texture_slot_diag_hash_cache = CalculateHash();
    }
    std::scoped_lock lock{g_texture_slot_variance_mutex};
    auto& slot_set = g_texture_types_seen[*texture_slot_diag_hash_cache]
                                          [MakeCbufKey(cbuf_index, cbuf_offset)];
    const bool newly_distinct = slot_set.insert(type).second;

    // Handle-specialized texture-key path; see CapturedPhase4PrototypeHandles.
    // comment in shader_environment.h. Independent of the newly_distinct/diagnostic logging
    // below: needs every resolved handle for this slot recorded, not just the ones that
    // introduce a new distinct value.
    if (Shader::IsPhase4PrototypeSlot(cbuf_index, cbuf_offset)) {
        phase4_prototype_handles.insert(handle);
    }

    // Fired only on a genuinely NEW distinct value for a slot that already had at least one —
    // i.e. exactly the "variance" events the aggregate histogram in
    // LogTextureSlotVarianceReportThrottled() counts, logged individually and immediately
    // (not throttled: real session data so far shows these are rare — order of 10, not
    // thousands — so there's no log-spam risk the way there would be for the per-instruction
    // recording this function does on every call). TextureType values are logged as their
    // raw underlying u32 rather than a name — see shader_info.h's `enum class TextureType`
    // for the mapping (0=Color1D, 1=ColorArray1D, 2=Color2D, 3=ColorArray2D, 4=Color3D,
    // 5=ColorCube, 6=ColorArrayCube, 7=Buffer, 8=Color2DRect) — no existing fmt::formatter
    // for this enum was found to reuse, and guessing at one risked a compile error this
    // session has no way to catch.
    if (newly_distinct && slot_set.size() > 1) {
        std::string values;
        for (const Shader::TextureType seen : slot_set) {
            values += fmt::format("{}{}", values.empty() ? "" : ",",
                                   static_cast<u32>(seen));
        }
        LOG_INFO(Render_Vulkan,
                 "Texture slot variance (TextureType) NEW distinct value: shader unique_hash="
                 "{:016x} cbuf_index={} cbuf_offset={} newly observed type={} — slot's full "
                 "distinct set so far: [{}]",
                 *texture_slot_diag_hash_cache, cbuf_index, cbuf_offset,
                 static_cast<u32>(type), values);

        // Adaptive slot learning (revised):
        // real-world reasoning this responds to). This IS the trigger — a coordinate that
        // just proved it has real variance, and isn't already active (an already-active
        // slot's own translation-time canonicalization means its variance never reaches this
        // branch as a genuinely new distinct value in the first place, so the
        // IsPhase4PrototypeSlot check below is a belt-and-suspenders guard, not the primary
        // reason this rarely double-fires). Two things happen, not one:
        //
        // 1. Grows THIS session's active table immediately — not deferred to next session.
        // Once a shader's variants are BOTH in the driver's persisted VkPipelineCache blob
        // (vulkan.bin), this does nothing further for that shader; the
        // driver-level cache already skips recompiling it regardless. What immediate growth
        // actually buys is every OTHER shader referencing the same coordinate that hasn't
        // been translated yet THIS session (real for TotK: ~15 shaders share the
        // (2,192)/(2,280) pattern) — each gets the polymorphic treatment from its own first
        // translation instead of independently rediscovering the same variance later. Safe
        // to do live, unlike touching an ALREADY-translated shader: this only ever appends
        // past the end of the current table (MergePhase4PrototypeSlots preserves existing
        // entries in their existing order, then appends anything new), so no
        // already-handed-out SpecId or bit position ever moves mid-session — see
        // Shader::SetActivePhase4PrototypeSlots's doc comment (environment.h) for the full
        // stability argument.
        //
        // 2. Still records to g_phase4_prototype_candidates for cross-session persistence —
        // covers what (1) doesn't: recovering the benefit after vulkan.bin itself gets
        // invalidated (a GPU driver update, a manual wipe, a fresh install) without
        // re-discovering variance the slow way session after session.
        //
        // Both under the lock already held above: growing the table is a
        // read-current/append/publish sequence, and doing it outside this lock would let two
        // threads both read the same "current" table and each publish their own
        // one-larger version, silently discarding whichever published first for the rest of
        // this session — never corrupts anything, but a needless lost learning opportunity.
        if (!Shader::IsPhase4PrototypeSlot(cbuf_index, cbuf_offset)) {
            const Shader::Phase4PrototypeSlot new_slot{.cbuf_index = cbuf_index,
                                                        .cbuf_offset = cbuf_offset};
            const bool already_recorded{std::ranges::any_of(
                g_phase4_prototype_candidates,
                [cbuf_index, cbuf_offset](const Shader::Phase4PrototypeSlot& slot) {
                    return slot.cbuf_index == cbuf_index && slot.cbuf_offset == cbuf_offset;
                })};
            if (!already_recorded) {
                g_phase4_prototype_candidates.push_back(new_slot);
            }

            const std::span<const Shader::Phase4PrototypeSlot> current{
                Shader::ActivePhase4PrototypeSlots()};
            std::vector<Shader::Phase4PrototypeSlot> grown{current.begin(), current.end()};
            grown.push_back(new_slot);
            Shader::SetActivePhase4PrototypeSlots(Shader::MergePhase4PrototypeSlots(grown));
        }
    }
}


void GenericEnvironment::RecordResolvedTexturePixelFormat(const Shader::TextureSlot& slot,
                                                           u32 handle,
                                                           Shader::TexturePixelFormat format) {
    logical_texture_slots[slot].pixel_format = format;
    logical_texture_handles.insert(handle);
    const u32 cbuf_index{slot.cbuf_index};
    const u32 cbuf_offset{slot.cbuf_offset};
    if (!texture_slot_diag_hash_cache) {
        texture_slot_diag_hash_cache = CalculateHash();
    }
    std::scoped_lock lock{g_texture_slot_variance_mutex};
    g_pixel_formats_seen[*texture_slot_diag_hash_cache][MakeCbufKey(cbuf_index, cbuf_offset)]
        .insert(format);
}

void GenericEnvironment::RecordResolvedIsTexturePixelFormatInteger(const Shader::TextureSlot& slot,
                                                                    bool is_integer) {
    logical_texture_slots[slot].is_integer = is_integer;
    const u32 cbuf_index{slot.cbuf_index};
    const u32 cbuf_offset{slot.cbuf_offset};
    if (!texture_slot_diag_hash_cache) {
        texture_slot_diag_hash_cache = CalculateHash();
    }
    std::scoped_lock lock{g_texture_slot_variance_mutex};
    g_is_integer_seen[*texture_slot_diag_hash_cache][MakeCbufKey(cbuf_index, cbuf_offset)]
        .insert(is_integer);
}

void GenericEnvironment::LogTextureSlotVarianceReportThrottled() {
    if (g_report_call_count.fetch_add(1) % kLogEveryNCalls != 0) {
        return;
    }
    std::scoped_lock lock{g_texture_slot_variance_mutex};
    // Histogram: how many (shader, slot) pairs saw exactly N distinct values, N = 1..5+. A
    // slot that only ever saw 1 distinct value needs no specialization at all — its current
    // single fixed OpTypeImage is already correct for every real draw seen so far.
    //
    // A cache HIT is never observed here, and does not need to be: a hit only happens when
    // texture_key already matches a stored entry, and texture_key is derived from the
    // resolved type/format (ComputeTextureKey, spirv_cache.cpp) — so a hit is, by
    // construction, always a repeat of an already-recorded value, never a new one. Only
    // translations (== misses) can introduce a new distinct value, and only translations
    // reach this recorder, so the histogram is sound without separately instrumenting hits.
    //
    // CAVEAT — read before treating the PixelFormat report as equally actionable to the
    // TextureType report: for ordinary SAMPLED textures (Info::texture_descriptors), pixel
    // format never becomes part of the emitted OpTypeImage at all (spirv_emit_context.cpp's
    // ImageType(EmitContext&, const TextureDescriptor&) hardcodes spv::ImageFormat::Unknown
    // unconditionally — TextureDescriptor has no format field to begin with, see
    // shader_info.h). Pixel-format variance only has SPIR-V-correctness consequences for
    // STORAGE images (Info::image_descriptors, which DOES carry ImageFormat and DOES bake it
    // into OpTypeImage), plus one narrow unrelated case (the !support_snorm_render_buffer
    // texel-fetch workaround, texture_pass.cpp). This instrumentation does not distinguish
    // which category a given slot resolves to — the counts below mix both. A high distinct
    // PixelFormat count is not by itself evidence that specialization constants are needed;
    // cross-reference against whether that same slot's instructions are ever ImageRead/
    // ImageWrite/ImageAtomic (storage) before treating it as load-bearing. IsInteger below
    // has no equivalent caveat — component type is baked into OpTypeImage's Sampled Type
    // operand for BOTH sampled and storage images (ImageType, spirv_emit_context.cpp), so any
    // variance it reports is directly actionable without needing this same cross-reference.
    auto summarize = [](const auto& seen_map, const char* label) {
        std::array<size_t, 5> bucket{}; // bucket[4] == "5 or more"
        size_t total_slots = 0;
        size_t max_distinct = 0;
        for (const auto& [unique_hash, slots] : seen_map) {
            for (const auto& [slot_key, values] : slots) {
                ++total_slots;
                const size_t distinct = values.size();
                max_distinct = std::max(max_distinct, distinct);
                bucket[std::min<size_t>(distinct, 5) - 1]++;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "Texture slot variance ({}): {} distinct (shader, slot) pairs observed so "
                 "far — {} saw exactly 1 distinct value, {} saw 2, {} saw 3, {} saw 4, {} saw "
                 "5+. Max distinct value count for any single slot: {}.",
                 label, total_slots, bucket[0], bucket[1], bucket[2], bucket[3], bucket[4],
                 max_distinct);
    };
    summarize(g_texture_types_seen, "TextureType");
    summarize(g_pixel_formats_seen, "TexturePixelFormat, sampled+storage mixed -- see caveat above");
    summarize(g_is_integer_seen, "IsTexturePixelFormatInteger");
}

void GenericEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

void GenericEnvironment::Serialize(std::ostream& file) const {
    const u64 code_size{static_cast<u64>(CachedSizeBytes())};
    const u64 num_texture_types{static_cast<u64>(texture_types.size())};
    const u64 num_texture_pixel_formats{static_cast<u64>(texture_pixel_formats.size())};
    const u64 num_logical_texture_slots{static_cast<u64>(logical_texture_slots.size())};
    const u64 num_logical_texture_handles{static_cast<u64>(logical_texture_handles.size())};
    const u64 num_cbuf_values{static_cast<u64>(cbuf_values.size())};
    const u64 num_texture_handle_cbuf_keys{static_cast<u64>(texture_handle_cbuf_keys.size())};
    const u64 num_cbuf_replacement_values{static_cast<u64>(cbuf_replacements.size())};
    const u64 num_cbuf_sizes{static_cast<u64>(cbuf_sizes.size())};
    // Replay corpus must be byte-stable. Unordered-map iteration otherwise makes
    // two equivalent captured environments produce different files, defeating
    // reproducibility and corrupt-file/shuffle tests.
    const auto sorted_entries = [](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        std::vector<const Entry*> sorted;
        sorted.reserve(entries.size());
        for (const Entry& entry : entries) {
            sorted.push_back(&entry);
        }
        std::ranges::sort(sorted, {}, [](const Entry* entry) -> const auto& {
            return entry->first;
        });
        return sorted;
    };
    const auto sorted_texture_types{sorted_entries(texture_types)};
    const auto sorted_texture_pixel_formats{sorted_entries(texture_pixel_formats)};
    const auto sorted_cbuf_values{sorted_entries(cbuf_values)};
    const auto sorted_cbuf_replacements{sorted_entries(cbuf_replacements)};
    const auto sorted_cbuf_sizes{sorted_entries(cbuf_sizes)};
    std::vector<std::pair<Shader::TextureSlot, Shader::TextureSlotShape>> sorted_logical_slots{
        logical_texture_slots.begin(), logical_texture_slots.end()};
    std::ranges::sort(sorted_logical_slots, {},
                      &std::pair<Shader::TextureSlot, Shader::TextureSlotShape>::first);
    std::vector<u32> sorted_logical_handles{logical_texture_handles.begin(),
                                            logical_texture_handles.end()};
    std::ranges::sort(sorted_logical_handles);
    std::vector<u64> sorted_texture_handle_cbuf_keys{texture_handle_cbuf_keys.begin(),
                                                      texture_handle_cbuf_keys.end()};
    std::ranges::sort(sorted_texture_handle_cbuf_keys);

    file.write(reinterpret_cast<const char*>(&code_size), sizeof(code_size))
        .write(reinterpret_cast<const char*>(&num_texture_types), sizeof(num_texture_types))
        .write(reinterpret_cast<const char*>(&num_texture_pixel_formats),
               sizeof(num_texture_pixel_formats))
        .write(reinterpret_cast<const char*>(&num_logical_texture_slots),
               sizeof(num_logical_texture_slots))
        .write(reinterpret_cast<const char*>(&num_logical_texture_handles),
               sizeof(num_logical_texture_handles))
        .write(reinterpret_cast<const char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .write(reinterpret_cast<const char*>(&num_texture_handle_cbuf_keys),
               sizeof(num_texture_handle_cbuf_keys))
        .write(reinterpret_cast<const char*>(&num_cbuf_replacement_values),
               sizeof(num_cbuf_replacement_values))
        .write(reinterpret_cast<const char*>(&num_cbuf_sizes), sizeof(num_cbuf_sizes))
        .write(reinterpret_cast<const char*>(&local_memory_size), sizeof(local_memory_size))
        .write(reinterpret_cast<const char*>(&texture_bound), sizeof(texture_bound))
        .write(reinterpret_cast<const char*>(&start_address), sizeof(start_address))
        .write(reinterpret_cast<const char*>(&cached_lowest), sizeof(cached_lowest))
        .write(reinterpret_cast<const char*>(&cached_highest), sizeof(cached_highest))
        .write(reinterpret_cast<const char*>(&viewport_transform_state),
               sizeof(viewport_transform_state))
        .write(reinterpret_cast<const char*>(&stage), sizeof(stage))
        .write(reinterpret_cast<const char*>(code.data()), code_size);
    for (const auto* entry : sorted_texture_types) {
        const auto& [key, type] = *entry;
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto* entry : sorted_texture_pixel_formats) {
        const auto& [key, format] = *entry;
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&format), sizeof(format));
    }
    for (const auto& [slot, shape] : sorted_logical_slots) {
        const u8 has_type{shape.type.has_value()};
        const u8 has_format{shape.pixel_format.has_value()};
        const u8 has_integer{shape.is_integer.has_value()};
        const u8 integer_value{shape.is_integer.value_or(false)};
        const u8 has_secondary{slot.has_secondary};
        file.write(reinterpret_cast<const char*>(&slot.cbuf_index), sizeof(slot.cbuf_index))
            .write(reinterpret_cast<const char*>(&slot.cbuf_offset), sizeof(slot.cbuf_offset))
            .write(reinterpret_cast<const char*>(&slot.shift_left), sizeof(slot.shift_left))
            .write(reinterpret_cast<const char*>(&slot.secondary_cbuf_index),
                   sizeof(slot.secondary_cbuf_index))
            .write(reinterpret_cast<const char*>(&slot.secondary_cbuf_offset),
                   sizeof(slot.secondary_cbuf_offset))
            .write(reinterpret_cast<const char*>(&slot.secondary_shift_left),
                   sizeof(slot.secondary_shift_left))
            .write(reinterpret_cast<const char*>(&slot.count), sizeof(slot.count))
            .write(reinterpret_cast<const char*>(&has_secondary), sizeof(has_secondary))
            .write(reinterpret_cast<const char*>(&has_type), sizeof(has_type))
            .write(reinterpret_cast<const char*>(&has_format), sizeof(has_format))
            .write(reinterpret_cast<const char*>(&has_integer), sizeof(has_integer))
            .write(reinterpret_cast<const char*>(&integer_value), sizeof(integer_value));
        if (shape.type) {
            file.write(reinterpret_cast<const char*>(&*shape.type), sizeof(*shape.type));
        }
        if (shape.pixel_format) {
            file.write(reinterpret_cast<const char*>(&*shape.pixel_format), sizeof(*shape.pixel_format));
        }
    }
    for (const u32 handle : sorted_logical_handles) {
        file.write(reinterpret_cast<const char*>(&handle), sizeof(handle));
    }
    for (const auto* entry : sorted_cbuf_values) {
        const auto& [key, type] = *entry;
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    // Just the keys — no associated value, since this is a set (see
    // ReadCbufValueForTextureHandle's doc comment in environment.h). Written as its
    // own count + loop rather than piggybacking on cbuf_values above so a reader can
    // tell which of cbuf_values' entries are texture-handle reads without needing to
    // cross-reference anything beyond this one extra list.
    for (const u64 key : sorted_texture_handle_cbuf_keys) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key));
    }
    for (const auto* entry : sorted_cbuf_replacements) {
        const auto& [key, type] = *entry;
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&type), sizeof(type));
    }
    for (const auto* entry : sorted_cbuf_sizes) {
        const auto& [key, size] = *entry;
        file.write(reinterpret_cast<const char*>(&key), sizeof(key))
            .write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    if (stage == Shader::Stage::Compute) {
        file.write(reinterpret_cast<const char*>(&workgroup_size), sizeof(workgroup_size))
            .write(reinterpret_cast<const char*>(&shared_memory_size), sizeof(shared_memory_size));
    } else {
        file.write(reinterpret_cast<const char*>(&sph), sizeof(sph));
        if (stage == Shader::Stage::Geometry) {
            file.write(reinterpret_cast<const char*>(&gp_passthrough_mask),
                       sizeof(gp_passthrough_mask));
        }
    }
}

std::optional<u64> GenericEnvironment::TryFindSize() {
    static constexpr size_t BLOCK_SIZE = 0x1000;
    static constexpr size_t MAXIMUM_SIZE = 0x100000;

    GPUVAddr guest_addr{program_base + start_address};
    size_t offset{0};
    size_t size{BLOCK_SIZE};
    while (size <= MAXIMUM_SIZE) {
        code.resize(size / INST_SIZE);
        u64* const data = code.data() + offset / INST_SIZE;
        gpu_memory->ReadBlock(guest_addr, data, BLOCK_SIZE);
        for (size_t index = 0; index < BLOCK_SIZE; index += INST_SIZE) {
            const u64 inst = data[index / INST_SIZE];
            if (inst == MAXWELL_SELF_BRANCH_A || inst == MAXWELL_SELF_BRANCH_B) {
                return offset + index;
            }
        }
        guest_addr += BLOCK_SIZE;
        size += BLOCK_SIZE;
        offset += BLOCK_SIZE;
    }
    return std::nullopt;
}

Tegra::Texture::TICEntry GenericEnvironment::ReadTextureInfo(GPUVAddr tic_addr, u32 tic_limit,
                                                             bool via_header_index, u32 raw) {
    const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
    if (handle.first > tic_limit) {
        // Common sentinel values that games use to indicate "no texture" or "unbound texture"
        // 0xfffffff8 = -8 (signed), commonly used as a sentinel value
        constexpr u32 COMMON_SENTINEL_VALUES[] = {0xfffffff8, 0xffffffff};
        const bool is_sentinel = std::find(std::begin(COMMON_SENTINEL_VALUES),
                                           std::end(COMMON_SENTINEL_VALUES), raw) !=
                                 std::end(COMMON_SENTINEL_VALUES);

        // Log each unique invalid handle only once to reduce spam
        static std::unordered_set<u32> logged_handles;
        const bool already_logged = logged_handles.contains(raw);

        if (!already_logged) {
            logged_handles.insert(raw);
            if (is_sentinel) {
                // Sentinel values are expected and not errors, use DEBUG level
                LOG_DEBUG(HW_GPU,
                          "Texture handle sentinel value detected (likely unbound texture). "
                          "Raw handle: 0x{:08x}, via_header_index: {}",
                          raw, via_header_index);
            } else {
                // Unexpected invalid handles are warnings
                LOG_WARNING(HW_GPU,
                            "Texture handle index {} exceeds TIC limit {}, clamping to valid range. "
                            "Raw handle: 0x{:08x}, via_header_index: {}",
                            handle.first, tic_limit, raw, via_header_index);
            }
        }

        // Return a default TICEntry with a safe fallback format
        Tegra::Texture::TICEntry entry{};
        // Set to a known safe format (A8B8G8R8_UNORM) using Assign method
        entry.format.Assign(Tegra::Texture::TextureFormat::A8B8G8R8);
        entry.r_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.g_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.b_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.a_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.texture_type.Assign(Tegra::Texture::TextureType::Texture2D);
        return entry;
    }
    const GPUVAddr descriptor_addr{tic_addr + handle.first * sizeof(Tegra::Texture::TICEntry)};
    Tegra::Texture::TICEntry entry;
    gpu_memory->ReadBlock(descriptor_addr, &entry, sizeof(entry));
    return entry;
}

// Resolve a texture type before an Environment exists, using ReadTextureInfo's fallback.
Shader::TextureType ResolveTextureTypeFromRawHandle(Tegra::MemoryManager& gpu_memory,
                                                     GPUVAddr tic_addr, u32 tic_limit,
                                                     bool via_header_index, u32 raw) {
    const auto handle{Tegra::Texture::TexturePair(raw, via_header_index)};
    Tegra::Texture::TICEntry entry;
    if (handle.first > tic_limit) {
        // Keep the normal safe fallback without per-draw sentinel logging.
        entry = Tegra::Texture::TICEntry{};
        entry.format.Assign(Tegra::Texture::TextureFormat::A8B8G8R8);
        entry.r_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.g_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.b_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.a_type.Assign(Tegra::Texture::ComponentType::UNORM);
        entry.texture_type.Assign(Tegra::Texture::TextureType::Texture2D);
    } else {
        const GPUVAddr descriptor_addr{tic_addr +
                                       handle.first * sizeof(Tegra::Texture::TICEntry)};
        gpu_memory.ReadBlock(descriptor_addr, &entry, sizeof(entry));
    }
    return ConvertTextureType(entry);
}

GraphicsEnvironment::GraphicsEnvironment(Tegra::Engines::Maxwell3D& maxwell3d_,
                                         Tegra::MemoryManager& gpu_memory_,
                                         Tegra::Engines::Maxwell3D::Regs::ShaderType program, GPUVAddr program_base_,
                                         u32 start_address_)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, maxwell3d{&maxwell3d_} {
    gpu_memory->ReadBlock(program_base + start_address, &sph, sizeof(sph));
    initial_offset = sizeof(sph);
    gp_passthrough_mask = maxwell3d->regs.post_vtg_shader_attrib_skip_mask;
    switch (program) {
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::VertexA:
        stage = Shader::Stage::VertexA;
        stage_index = 0;
        break;
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::VertexB:
        stage = Shader::Stage::VertexB;
        stage_index = 0;
        break;
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::TessellationInit:
        stage = Shader::Stage::TessellationControl;
        stage_index = 1;
        break;
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::Tessellation:
        stage = Shader::Stage::TessellationEval;
        stage_index = 2;
        break;
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::Geometry:
        stage = Shader::Stage::Geometry;
        stage_index = 3;
        break;
    case Tegra::Engines::Maxwell3D::Regs::ShaderType::Pixel:
        stage = Shader::Stage::Fragment;
        stage_index = 4;
        break;
    default:
        ASSERT_MSG(false, "Invalid program={}", program);
        break;
    }
    const u64 local_size{sph.LocalMemorySize()};
    ASSERT(local_size <= std::numeric_limits<u32>::max());
    local_memory_size = static_cast<u32>(local_size) + sph.common3.shader_local_memory_crs_size;
    texture_bound = maxwell3d->regs.bindless_texture_const_buffer_slot;
    is_proprietary_driver = texture_bound == 2;
    has_hle_engine_state =
        maxwell3d->engine_state == Tegra::Engines::Maxwell3D::EngineHint::OnHLEMacro;
    // Pre-capture cbuf sizes on the main thread while Maxwell3D state is live.
    // ReadCbufSize() is called later on async shader compiler worker threads by which
    // point state.shader_stages may have changed and cbuf.enabled may be false,
    // causing a spurious 0 that would inflate descriptor array sizes via the fallback.
    const auto& stage_cbufs{maxwell3d->state.shader_stages[stage_index].const_buffers};
    for (u32 i = 0; i < static_cast<u32>(stage_cbufs.size()); ++i) {
        if (stage_cbufs[i].enabled) {
            cbuf_sizes.emplace(i, static_cast<u32>(stage_cbufs[i].size));
        }
    }
}

u32 GraphicsEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto& cbuf{maxwell3d->state.shader_stages[stage_index].const_buffers[cbuf_index]};
    ASSERT(cbuf.enabled);
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        value = gpu_memory->Read<u32>(cbuf.address + cbuf_offset);
    }
    const u64 key = MakeCbufKey(cbuf_index, cbuf_offset);
    cbuf_values.emplace(key, value);
    RecordCbufRead(key);
    return value;
}

u32 GraphicsEnvironment::ReadCbufSize(u32 cbuf_index) {
    // Sizes were pre-captured in the constructor on the main thread.
    const auto it{cbuf_sizes.find(cbuf_index)};
    return it != cbuf_sizes.end() ? it->second : 0;
}

std::optional<Shader::ReplaceConstant> GraphicsEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                                  u32 offset) {
    if (!has_hle_engine_state) {
        return std::nullopt;
    }
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = maxwell3d->replace_table.find(key);
    if (it == maxwell3d->replace_table.end()) {
        return std::nullopt;
    }
    const auto converted_value = [](Tegra::Engines::Maxwell3D::HLEReplacementAttributeType name) {
        switch (name) {
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseVertex:
            return Shader::ReplaceConstant::BaseVertex;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::BaseInstance:
            return Shader::ReplaceConstant::BaseInstance;
        case Tegra::Engines::Maxwell3D::HLEReplacementAttributeType::DrawID:
            return Shader::ReplaceConstant::DrawID;
        default:
            UNREACHABLE();
        }
    }(it->second);
    cbuf_replacements.emplace(key, converted_value);
    return converted_value;
}

Shader::TextureType GraphicsEnvironment::ReadTextureType(u32 handle) {
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat GraphicsEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding};
    auto entry =
        ReadTextureInfo(regs.tex_header.Address(), regs.tex_header.limit, via_header_index, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool GraphicsEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 GraphicsEnvironment::ReadViewportTransformState() {
    const auto& regs{maxwell3d->regs};
    viewport_transform_state = regs.viewport_scale_offset_enabled;
    return viewport_transform_state;
}

ComputeEnvironment::ComputeEnvironment(Tegra::Engines::KeplerCompute& kepler_compute_,
                                       Tegra::MemoryManager& gpu_memory_, GPUVAddr program_base_,
                                       u32 start_address_)
    : GenericEnvironment{gpu_memory_, program_base_, start_address_}, kepler_compute{
                                                                          &kepler_compute_} {
    const auto& qmd{kepler_compute->launch_description};
    stage = Shader::Stage::Compute;
    local_memory_size = qmd.local_pos_alloc + qmd.local_crs_alloc;
    texture_bound = kepler_compute->regs.tex_cb_index;
    is_proprietary_driver = texture_bound == 2;
    shared_memory_size = qmd.shared_alloc;
    workgroup_size = {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z};
    // Pre-capture cbuf sizes on the main thread while launch_description is live.
    for (u32 i = 0; i < static_cast<u32>(qmd.const_buffer_config.size()); ++i) {
        if ((qmd.const_buffer_enable_mask.Value() >> i) & 1) {
            cbuf_sizes.emplace(i, static_cast<u32>(qmd.const_buffer_config[i].size));
        }
    }
}

u32 ComputeEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto& qmd{kepler_compute->launch_description};
    ASSERT(((qmd.const_buffer_enable_mask.Value() >> cbuf_index) & 1) != 0);
    const auto& cbuf{qmd.const_buffer_config[cbuf_index]};
    u32 value{};
    if (cbuf_offset < cbuf.size) {
        value = gpu_memory->Read<u32>(cbuf.Address() + cbuf_offset);
    }
    const u64 key = MakeCbufKey(cbuf_index, cbuf_offset);
    cbuf_values.emplace(key, value);
    RecordCbufRead(key);
    return value;
}

u32 ComputeEnvironment::ReadCbufSize(u32 cbuf_index) {
    // Sizes were pre-captured in the constructor on the main thread.
    const auto it{cbuf_sizes.find(cbuf_index)};
    return it != cbuf_sizes.end() ? it->second : 0;
}

Shader::TextureType ComputeEnvironment::ReadTextureType(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TextureType result{ConvertTextureType(entry)};
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat ComputeEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto& regs{kepler_compute->regs};
    const auto& qmd{kepler_compute->launch_description};
    auto entry = ReadTextureInfo(regs.tic.Address(), regs.tic.limit, qmd.linked_tsc != 0, handle);
    const Shader::TexturePixelFormat result(ConvertTexturePixelFormat(entry));
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool ComputeEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 ComputeEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

void FileEnvironment::Deserialize(std::ifstream& file) {
    u64 code_size{};
    u64 num_texture_types{};
    u64 num_texture_pixel_formats{};
    u64 num_logical_texture_slots{};
    u64 num_logical_texture_handles{};
    u64 num_cbuf_values{};
    u64 num_texture_handle_cbuf_keys{};
    u64 num_cbuf_replacement_values{};
    u64 num_cbuf_sizes{};
    file.read(reinterpret_cast<char*>(&code_size), sizeof(code_size))
        .read(reinterpret_cast<char*>(&num_texture_types), sizeof(num_texture_types))
        .read(reinterpret_cast<char*>(&num_texture_pixel_formats),
              sizeof(num_texture_pixel_formats))
        .read(reinterpret_cast<char*>(&num_logical_texture_slots),
              sizeof(num_logical_texture_slots))
        .read(reinterpret_cast<char*>(&num_logical_texture_handles),
              sizeof(num_logical_texture_handles))
        .read(reinterpret_cast<char*>(&num_cbuf_values), sizeof(num_cbuf_values))
        .read(reinterpret_cast<char*>(&num_texture_handle_cbuf_keys),
              sizeof(num_texture_handle_cbuf_keys))
        .read(reinterpret_cast<char*>(&num_cbuf_replacement_values),
              sizeof(num_cbuf_replacement_values))
        .read(reinterpret_cast<char*>(&num_cbuf_sizes), sizeof(num_cbuf_sizes))
        .read(reinterpret_cast<char*>(&local_memory_size), sizeof(local_memory_size))
        .read(reinterpret_cast<char*>(&texture_bound), sizeof(texture_bound))
        .read(reinterpret_cast<char*>(&start_address), sizeof(start_address))
        .read(reinterpret_cast<char*>(&read_lowest), sizeof(read_lowest))
        .read(reinterpret_cast<char*>(&read_highest), sizeof(read_highest))
        .read(reinterpret_cast<char*>(&viewport_transform_state), sizeof(viewport_transform_state))
        .read(reinterpret_cast<char*>(&stage), sizeof(stage));
    const u64 read_span_size = read_highest >= read_lowest
                                   ? static_cast<u64>(read_highest) - read_lowest + sizeof(u64)
                                   : 0;
    if (code_size == 0 || code_size > MAX_SERIALIZED_SHADER_BYTES ||
        code_size % sizeof(u64) != 0 || read_span_size == 0 ||
        read_span_size > code_size || (read_highest - read_lowest) % sizeof(u64) != 0 ||
        num_texture_types > MAX_SERIALIZED_ENV_ENTRIES ||
        num_texture_pixel_formats > MAX_SERIALIZED_ENV_ENTRIES ||
        num_logical_texture_slots > MAX_SERIALIZED_ENV_ENTRIES ||
        num_logical_texture_handles > MAX_SERIALIZED_ENV_ENTRIES ||
        num_cbuf_values > MAX_SERIALIZED_ENV_ENTRIES ||
        num_texture_handle_cbuf_keys > MAX_SERIALIZED_ENV_ENTRIES ||
        num_cbuf_replacement_values > MAX_SERIALIZED_ENV_ENTRIES ||
        num_cbuf_sizes > MAX_SERIALIZED_ENV_ENTRIES ||
        static_cast<u32>(stage) > static_cast<u32>(Shader::Stage::VertexA)) {
        throw std::ios_base::failure{"invalid serialized shader environment size or stage"};
    }
    code.resize(Common::DivCeil(code_size, sizeof(u64)));
    file.read(reinterpret_cast<char*>(code.data()), code_size);
    for (size_t i = 0; i < num_texture_types; ++i) {
        u32 key;
        Shader::TextureType type;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&type), sizeof(type));
        if (!IsValidTextureType(type)) {
            throw std::ios_base::failure{"invalid serialized texture type"};
        }
        if (!texture_types.emplace(key, type).second) {
            throw std::ios_base::failure{"duplicate serialized texture type"};
        }
    }
    for (size_t i = 0; i < num_texture_pixel_formats; ++i) {
        u32 key;
        Shader::TexturePixelFormat format;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&format), sizeof(format));
        if (!IsValidTexturePixelFormat(format)) {
            throw std::ios_base::failure{"invalid serialized texture pixel format"};
        }
        if (!texture_pixel_formats.emplace(key, format).second) {
            throw std::ios_base::failure{"duplicate serialized texture pixel format"};
        }
    }
    for (size_t i = 0; i < num_logical_texture_slots; ++i) {
        Shader::TextureSlot slot{};
        u8 has_secondary{}, has_type{}, has_format{}, has_integer{}, integer_value{};
        file.read(reinterpret_cast<char*>(&slot.cbuf_index), sizeof(slot.cbuf_index))
            .read(reinterpret_cast<char*>(&slot.cbuf_offset), sizeof(slot.cbuf_offset))
            .read(reinterpret_cast<char*>(&slot.shift_left), sizeof(slot.shift_left))
            .read(reinterpret_cast<char*>(&slot.secondary_cbuf_index),
                  sizeof(slot.secondary_cbuf_index))
            .read(reinterpret_cast<char*>(&slot.secondary_cbuf_offset),
                  sizeof(slot.secondary_cbuf_offset))
            .read(reinterpret_cast<char*>(&slot.secondary_shift_left),
                  sizeof(slot.secondary_shift_left))
            .read(reinterpret_cast<char*>(&slot.count), sizeof(slot.count))
            .read(reinterpret_cast<char*>(&has_secondary), sizeof(has_secondary))
            .read(reinterpret_cast<char*>(&has_type), sizeof(has_type))
            .read(reinterpret_cast<char*>(&has_format), sizeof(has_format))
            .read(reinterpret_cast<char*>(&has_integer), sizeof(has_integer))
            .read(reinterpret_cast<char*>(&integer_value), sizeof(integer_value));
        // Primary-only slots retain secondary fields as mirrors of the primary
        // address. They are not required to be zero when has_secondary is false.
        if (has_secondary > 1 || has_type > 1 || has_format > 1 || has_integer > 1 ||
            integer_value > 1 || slot.shift_left > 31 || slot.secondary_shift_left > 31 ||
            slot.count == 0) {
            throw std::ios_base::failure{"invalid logical texture slot"};
        }
        slot.has_secondary = has_secondary != 0;
        Shader::TextureSlotShape shape{};
        if (has_type) {
            Shader::TextureType type{};
            file.read(reinterpret_cast<char*>(&type), sizeof(type));
            if (!IsValidTextureType(type)) {
                throw std::ios_base::failure{"invalid logical texture slot type"};
            }
            shape.type = type;
        }
        if (has_format) {
            Shader::TexturePixelFormat format{};
            file.read(reinterpret_cast<char*>(&format), sizeof(format));
            if (!IsValidTexturePixelFormat(format)) {
                throw std::ios_base::failure{"invalid logical texture slot pixel format"};
            }
            shape.pixel_format = format;
        }
        if (has_integer) {
            shape.is_integer = integer_value != 0;
        }
        if (!logical_texture_slots.emplace(slot, std::move(shape)).second) {
            throw std::ios_base::failure{"duplicate logical texture slot"};
        }
    }
    for (size_t i = 0; i < num_logical_texture_handles; ++i) {
        u32 handle{};
        file.read(reinterpret_cast<char*>(&handle), sizeof(handle));
        if (!logical_texture_handles.insert(handle).second) {
            throw std::ios_base::failure{"duplicate logical texture handle"};
        }
    }
    for (size_t i = 0; i < num_cbuf_values; ++i) {
        u64 key;
        u32 value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!cbuf_values.emplace(key, value).second) {
            throw std::ios_base::failure{"duplicate serialized cbuf value"};
        }
    }
    // Same set-of-keys-only format Serialize() wrote — see its doc comment. Read after
    // cbuf_values, matching write order exactly, since this format has no per-field
    // framing beyond the counts read up front.
    for (size_t i = 0; i < num_texture_handle_cbuf_keys; ++i) {
        u64 key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!texture_handle_cbuf_keys.insert(key).second) {
            throw std::ios_base::failure{"duplicate serialized texture-handle cbuf key"};
        }
    }
    for (size_t i = 0; i < num_cbuf_replacement_values; ++i) {
        u64 key;
        Shader::ReplaceConstant value;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!IsValidReplaceConstant(value)) {
            throw std::ios_base::failure{"invalid serialized cbuf replacement"};
        }
        if (!cbuf_replacements.emplace(key, value).second) {
            throw std::ios_base::failure{"duplicate serialized cbuf replacement"};
        }
    }
    for (size_t i = 0; i < num_cbuf_sizes; ++i) {
        u32 key;
        u32 size;
        file.read(reinterpret_cast<char*>(&key), sizeof(key))
            .read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!cbuf_sizes.emplace(key, size).second) {
            throw std::ios_base::failure{"duplicate serialized cbuf size"};
        }
    }
    if (stage == Shader::Stage::Compute) {
        file.read(reinterpret_cast<char*>(&workgroup_size), sizeof(workgroup_size))
            .read(reinterpret_cast<char*>(&shared_memory_size), sizeof(shared_memory_size));
        initial_offset = 0;
    } else {
        file.read(reinterpret_cast<char*>(&sph), sizeof(sph));
        initial_offset = sizeof(sph);
        if (stage == Shader::Stage::Geometry) {
            file.read(reinterpret_cast<char*>(&gp_passthrough_mask), sizeof(gp_passthrough_mask));
        }
    }
    is_proprietary_driver = texture_bound == 2;
}

void FileEnvironment::Dump(u64 pipeline_hash, u64 shader_hash) {
    DumpImpl(pipeline_hash, shader_hash, code, read_highest, read_lowest, initial_offset, stage);
}

u64 FileEnvironment::ReadInstruction(u32 address) {
    if (address < read_lowest || address > read_highest) {
        throw Shader::LogicError("Out of bounds address {}", address);
    }
    return code[(address - read_lowest) / sizeof(u64)];
}

bool FileEnvironment::HasValidEntryInstruction() const noexcept {
    return IsValidShaderEntryInstruction(initial_offset, start_address, read_lowest, read_highest,
                                         code);
}

std::optional<u64> FileEnvironment::ProgramIdentity() const noexcept {
    return ComputeMaxwellProgramIdentity(code);
}

u32 FileEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    const auto it{cbuf_values.find(MakeCbufKey(cbuf_index, cbuf_offset))};
    if (it == cbuf_values.end()) {
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

u32 FileEnvironment::ReadCbufSize(u32 cbuf_index) {
    const auto it{cbuf_sizes.find(cbuf_index)};
    return it != cbuf_sizes.end() ? it->second : 0;
}

Shader::TextureType FileEnvironment::ReadTextureType(u32 handle) {
    const auto it{texture_types.find(handle)};
    if (it == texture_types.end()) {
        throw Shader::LogicError("Uncached read texture type");
    }
    return it->second;
}

Shader::TexturePixelFormat FileEnvironment::ReadTexturePixelFormat(u32 handle) {
    const auto it{texture_pixel_formats.find(handle)};
    if (it == texture_pixel_formats.end()) {
        throw Shader::LogicError("Uncached read texture pixel format");
    }
    return it->second;
}

bool FileEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 FileEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

u32 FileEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 FileEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

u32 FileEnvironment::TextureBoundBuffer() const {
    return texture_bound;
}

std::array<u32, 3> FileEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

std::optional<Shader::ReplaceConstant> FileEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                              u32 offset) {
    const u64 key = (static_cast<u64>(bank) << 32) | static_cast<u64>(offset);
    auto it = cbuf_replacements.find(key);
    if (it == cbuf_replacements.end()) {
        return std::nullopt;
    }
    return it->second;
}

void SerializePipeline(std::span<const char> key, std::span<const GenericEnvironment* const> envs,
                       const std::filesystem::path& filename, u32 cache_version) try {
    if (!std::ranges::all_of(envs, &GenericEnvironment::CanBeSerialized)) {
        return;
    }

    // A writer can outlive a failed/aborted earlier load. Never append a new
    // schema record behind an old header: the next reader would otherwise see
    // a valid old version and decode the new bytes with the wrong layout.
    {
        std::ifstream existing(filename, std::ios::binary | std::ios::ate);
        if (existing.is_open()) {
            existing.exceptions(std::ifstream::failbit);
            const auto end{existing.tellg()};
            const auto end_offset{static_cast<std::streamoff>(end)};
            existing.seekg(0, std::ios::beg);
            std::array<char, 8> magic{};
            u32 existing_version{};
            if (end_offset < static_cast<std::streamoff>(magic.size() + sizeof(existing_version))) {
                throw std::ios_base::failure{"truncated pipeline cache header"};
            }
            existing.read(magic.data(), magic.size())
                .read(reinterpret_cast<char*>(&existing_version), sizeof(existing_version));
            if (magic != MAGIC_NUMBER || existing_version != cache_version) {
                existing.close();
                if (!Common::FS::RemoveFile(filename)) {
                    throw std::ios_base::failure{"failed to replace incompatible pipeline cache"};
                }
                LOG_INFO(Common_Filesystem, "Replacing incompatible pipeline cache file");
            }
        }
    }

    std::ofstream file(filename, std::ios::binary | std::ios::ate | std::ios::app);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }
    if (file.tellp() == 0) {
        // Write header
        file.write(MAGIC_NUMBER.data(), MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));
    }
    std::ostringstream record{std::ios::binary | std::ios::out};
    const u32 num_envs{static_cast<u32>(envs.size())};
    record.write(reinterpret_cast<const char*>(&num_envs), sizeof(num_envs));
    for (const GenericEnvironment* const env : envs) {
        env->Serialize(record);
    }
    record.write(key.data(), key.size_bytes());
    const std::string payload{record.str()};
    const u64 payload_size{static_cast<u64>(payload.size())};
    if (payload_size == 0 || payload_size > MAX_SERIALIZED_PIPELINE_RECORD_BYTES) {
        throw std::ios_base::failure{"invalid serialized pipeline record size"};
    }
    file.write(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size))
        .write(payload.data(), static_cast<std::streamsize>(payload.size()));

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

void LoadPipelines(
    std::stop_token stop_loading, const std::filesystem::path& filename, u32 expected_cache_version,
    Common::UniqueFunction<void, std::ifstream&, FileEnvironment> load_compute,
    Common::UniqueFunction<void, std::ifstream&, std::vector<FileEnvironment>> load_graphics) try {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return;
    }
    file.exceptions(std::ifstream::failbit);
    const auto end{file.tellg()};
    file.seekg(0, std::ios::beg);

    std::array<char, 8> magic_number;
    u32 cache_version;
    file.read(magic_number.data(), magic_number.size())
        .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
    if (magic_number != MAGIC_NUMBER || cache_version != expected_cache_version) {
        file.close();
        if (Common::FS::RemoveFile(filename)) {
            if (magic_number != MAGIC_NUMBER) {
                LOG_ERROR(Common_Filesystem, "Invalid pipeline cache file");
            }
            if (cache_version != expected_cache_version) {
                LOG_INFO(Common_Filesystem, "Deleting old pipeline cache");
            }
        } else {
            LOG_ERROR(Common_Filesystem,
                      "Invalid pipeline cache file and failed to delete it in \"{}\"",
                      Common::FS::PathToUTF8String(filename));
        }
        return;
    }
    while (file.tellg() != end) {
        if (stop_loading.stop_requested()) {
            return;
        }
        const auto record_size_offset{static_cast<std::streamoff>(file.tellg())};
        const auto end_offset{static_cast<std::streamoff>(end)};
        if (end_offset - record_size_offset < static_cast<std::streamoff>(sizeof(u64))) {
            LOG_WARNING(Common_Filesystem,
                        "Ignoring truncated trailing pipeline cache record header");
            return;
        }
        u64 record_size{};
        file.read(reinterpret_cast<char*>(&record_size), sizeof(record_size));
        const auto record_payload_offset{static_cast<std::streamoff>(file.tellg())};
        if (record_size == 0 || record_size > MAX_SERIALIZED_PIPELINE_RECORD_BYTES ||
            record_size > static_cast<u64>(end_offset - record_payload_offset)) {
            LOG_WARNING(Common_Filesystem,
                        "Ignoring truncated or oversized trailing pipeline cache record");
            return;
        }
        const std::streamoff record_end_offset{
            record_payload_offset + static_cast<std::streamoff>(record_size)};
        u32 num_envs{};
        file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));

        if (num_envs == 0 || num_envs > 64) {
            LOG_ERROR(Common_Filesystem, "Corrupted shader cache detected: num_envs={}", num_envs);
            throw std::ios_base::failure("Corrupted num_envs");
        }

        std::vector<FileEnvironment> envs(num_envs);
        for (FileEnvironment& env : envs) {
            env.Deserialize(file);
        }

        if (envs.front().ShaderStage() == Shader::Stage::Compute) {
            // Compute records have exactly one environment. Accepting trailing
            // graphics/duplicate environments here would silently discard their
            // captured contract while still consuming their bytes and key.
            if (envs.size() != 1) {
                throw std::ios_base::failure{"invalid multi-environment compute record"};
            }
            load_compute(file, std::move(envs.front()));
        } else {
            if (std::ranges::any_of(envs, [](const FileEnvironment& env) {
                    return env.ShaderStage() == Shader::Stage::Compute;
                })) {
                throw std::ios_base::failure{"compute environment in graphics record"};
            }
            load_graphics(file, std::move(envs));
        }
        if (static_cast<std::streamoff>(file.tellg()) != record_end_offset) {
            throw std::ios_base::failure{"pipeline cache record size mismatch"};
        }
    }

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    if (!Common::FS::RemoveFile(filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
    }
}

} // namespace VideoCommon
