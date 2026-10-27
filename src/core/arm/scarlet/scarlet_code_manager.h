// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The recompiler's header is a GNU C dialect that #defines short macro names (LIKELY, MP, YIELD,
// ...). Anything including this header must include it BEFORE other Citron headers.
#include <snx_recompiler.h>
#undef LIKELY
#undef UNLIKELY
#undef YIELD
#undef MP
#undef RX
#undef CBAR

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Core::Scarlet {

/// Where the manager gets guest code from. Implemented over the process's page table in Citron,
/// and over a plain buffer in the standalone harness (tools/scarlet-harness).
class CodeSource {
public:
    virtual ~CodeSource() = default;

    /// The contiguous executable guest range [lo, hi) around pc, 4-byte aligned, at most
    /// max_size bytes. False when pc is not executable.
    virtual bool ExecRangeAround(std::uint64_t pc, std::uint64_t max_size, std::uint64_t& lo,
                                 std::uint64_t& hi) = 0;
    virtual void Read(std::uint64_t addr, void* dst, std::size_t size) = 0;
};

/// All recompiled code of one guest process.
///
///  * The *primary* module is recompiled up front from the loaded NSO segments. Every CPU is bound
///    to it, so calls into it are plain dispatch-table loads.
///  * Everything else (NROs from ldr:ro, JIT-generated or later-mapped code) is *secondary*:
///    compiled on first call by RtAPI.miss with snx_recompile_from, as ScarletNX's own gnuabi
///    runtime does, and entered through a lookup table the manager owns. Generated code reaches it
///    through the miss path; calls from it into the primary module use the dispatch table.
///
/// Secondary modules are cached on disk by ScarletNX's module cache as well as held in memory.
/// Code of an invalidated function may still be running on another core, so invalidation clears
/// dispatch entries but retains native modules until process shutdown.
///
/// Thread safety: Miss() may run on every guest thread at once. Primary misses use ScarletNX's
/// worker-backed JIT; compiles for code outside the primary image are serialized here.
class CodeManager {
public:
    struct Segment {
        std::uint64_t addr;
        std::uint64_t size;
    };
    struct ModuleImage {
        std::uint64_t image_addr;
        std::uint64_t image_size;
        std::uint64_t text_offset;
        std::uint64_t text_size;
    };

    explicit CodeManager(CodeSource& source);
    ~CodeManager();
    CodeManager(const CodeManager&) = delete;
    CodeManager& operator=(const CodeManager&) = delete;

    /// Recompiles the primary module (blocking; cached on disk under cache_dir_utf8 by content
    /// hash). The owner then sets rt.svc, rt.undef, rt.brk and rt.poll on Rt() and binds CPUs.
    bool Initialize(const std::vector<Segment>& segments, const char* cache_dir_utf8,
                    std::string& error, std::function<void(std::size_t, std::size_t)> progress = {},
                    const std::vector<ModuleImage>& modules = {});

    /// The table every CPU of the process points at (CPU.rt). Its miss hook is the manager's.
    RtAPI& Rt() {
        return m_block.rt;
    }

    /// Points the CPU at the primary module and this manager's RtAPI. Needs cpu->mem set, and
    /// svc/undef/brk set on Rt().
    bool Bind(CPU* cpu);

    /// snx_recompiled_run on the primary module: runs from pc until cpu->pc == stop.
    void Run(CPU* cpu, std::uint64_t pc, std::uint64_t stop);
    /// Runs the code at pc from a copy compiled from current guest memory (stale primary code, or
    /// a primary miss in a changed line). False when pc cannot be compiled.
    [[nodiscard]] bool RunLiveFallback(CPU* cpu, std::uint64_t pc);

    /// Guest code in [addr, addr + size) may have changed (IC IVAU, mapping update, or debugger
    /// edit). Primary code is retired only for 64-byte lines whose bytes differ from what the
    /// primary module was compiled from, so permission flips and remaps of unchanged code cost a
    /// hash compare. Does not wait for running compiles. Safe to call from any thread.
    void Invalidate(std::uint64_t addr, std::uint64_t size);

    [[nodiscard]] std::size_t SecondaryModuleCount() const;

    [[nodiscard]] std::size_t SecondaryFunctionCount() const;

    /// Collects a runtime-discovered function entry point for future warm-start.
    void NoteHint(std::uint64_t pc);

    /// Saves collected runtime hints to disk for next-launch warm-start.
    void SaveHints(const char* cache_dir_utf8) const;

    /// Loads previously saved hints from disk.
    void LoadHints(const char* cache_dir_utf8);

    /// Pre-warms JIT units from loaded hints using library-jit.cache.
    void WarmUpHints();

    /// Saves hints to disk periodically if any new hints were added.
    void PeriodicSaveHints();

    /// Returns the current hint set (thread-safe copy).
    [[nodiscard]] std::vector<std::uint64_t> CopyHints() const;

private:
    struct RtBlock { // RtAPI first: a CPU.rt pointer converts back to the block
        RtAPI rt{};
        CodeManager* owner{};
    };

    using RangeList = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

    struct InvalidatedRangeContext {
        RangeList* ranges;
    };

    struct SecondaryModule {
        snx_recompiled* module{};
        std::uint64_t lo{};
        std::uint64_t hi{};
    };

    /// One bound CPU's view of the primary module's dispatch table (CPU.disp with its guest range).
    struct DispatchTable {
        GuestFn* table{};
        std::uint64_t lo{};
        std::uint64_t hi{};
    };

    static void MissThunk(CPU* c, std::uint64_t pc);
    static void CodeChangedThunk(CPU* c, std::uint64_t addr);
    static void NoteHintThunk(std::uint64_t pc);
    static int SkipThunk(void* context, std::uint64_t address);
    static void InvalidatedRangeThunk(void* context, std::uint64_t start, std::uint64_t end);

    void Miss(CPU* c, std::uint64_t pc);
    void CodeChanged(std::uint64_t addr);
    /// Refills CPU.disp slots in [lo, hi) with the manager's miss hook and returns how many it
    /// wrote. The AOT's stale guard (SNX_ABI 14) runs a stale function's entry through dispatch,
    /// so a slot that still names the retired function sends the guest back into that function: it
    /// looks the same slot up again and repeats its own body for ever. A slot that names the miss
    /// hook makes the call reach the queued JIT, which compiles the pc from live memory, re-arms
    /// the slot and resumes there.
    std::size_t RedirectPrimaryEntries(std::uint64_t lo, std::uint64_t hi);
    /// Rewrites the stored per-line hashes of primary code in [addr, end) with the hashes of the
    /// bytes there now, so an Invalidate that arrives again for the same bytes (the page table and
    /// the guest's own IC IVAU both report one patch) retires nothing a second time.
    void RefreshPrimaryHashes(std::uint64_t addr, std::uint64_t end);
    [[nodiscard]] bool IsInvalidated(std::uint64_t pc) const;
    void MergeInvalidatedRanges(RangeList ranges);
    [[nodiscard]] const RangeList* LoadInvalidatedRanges() const;
    /// Primary-segment runs [start, end) whose bytes differ from the compiled snapshot.
    [[nodiscard]] RangeList ChangedPrimaryRuns(std::uint64_t addr, std::uint64_t end) const;
    [[nodiscard]] bool InPrimarySegment(std::uint64_t address) const;
    [[nodiscard]] GuestFn FindSecondary(std::uint64_t pc) const;
    GuestFn CompileSecondary(std::uint64_t pc);
    void HandleCompilerOutputLine(const std::string& line);
    void LogCapturedCompilerOutput();

    CodeSource& m_source;
    RtBlock m_block{};
    void (*m_default_miss)(CPU*, std::uint64_t){}; // ScarletNX's queued, batched primary JIT

    snx_recompiler* m_recompiler{}; // primary (disk cache on)
    snx_recompiler* m_jit{};        // secondary (disk cache; code and guest addresses key entries)
    bool m_secondary_cache_enabled{};
    snx_recompiled* m_primary{};
    std::vector<Segment> m_segments;

    /// The dispatch table every bound CPU points at (cpu.disp, cpu.code_lo/code_hi), recorded at
    /// Bind time and deduplicated: every core of a process shares the primary module's one table.
    mutable std::mutex m_tables_lock;
    std::vector<DispatchTable> m_tables;
    bool m_load_staleness_checked{};

    /// Hash of every 64-byte line of each primary segment as compiled (parallel to m_segments).
    std::vector<std::vector<std::uint64_t>> m_segment_hashes;

    const std::uint64_t m_instance_id;
    std::mutex m_compile_lock; // snx_recompiler is one-thread-at-a-time; Invalidate never takes it
    std::mutex m_invalidate_lock; // serializes Invalidate calls
    mutable std::shared_mutex m_secondary_lock;
    std::unordered_map<std::uint64_t, GuestFn> m_secondary;
    std::vector<SecondaryModule> m_secondary_modules;
    /// Bumped by every Invalidate before it touches anything; a compile that sees it change
    /// re-verifies its source bytes before publishing.
    std::atomic<std::uint64_t> m_invalidate_gen{};

    /// Primary guest ranges whose compiled code is retired. Writers (under m_ranges_mutex) publish
    /// a new list and bump the generation; readers keep a per-thread copy keyed by it, so the
    /// hot IsInvalidated path takes no lock while nothing changes.
    mutable std::mutex m_ranges_mutex;
    std::shared_ptr<const RangeList> m_invalidated_ranges;
    std::atomic<std::uint64_t> m_ranges_generation{};
    std::atomic<bool> m_has_invalidations{};
    std::atomic<std::uint64_t> m_live_fallback_logs{};
    std::function<void(std::size_t, std::size_t)> m_recompile_progress;
    std::atomic<std::uint64_t> m_primary_miss_count{};
    std::atomic<std::uint64_t> m_primary_miss_time_ns{};
    std::ofstream m_compiler_trace;
    std::vector<std::string> m_pending_compiler_log_lines;

    mutable std::mutex m_hints_lock;
    std::vector<std::uint64_t> m_hints;
    std::unordered_set<std::uint64_t> m_hint_set;
    std::atomic<bool> m_hints_dirty{};
    std::string m_cache_dir;
};

} // namespace Core::Scarlet
