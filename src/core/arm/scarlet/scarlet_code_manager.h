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
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
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
    /// Recompile and run a stale primary instruction from current guest memory.
    [[nodiscard]] bool RunLiveFallback(CPU* cpu, std::uint64_t pc);

    /// Guest code in [addr, addr + size) changed (IC IVAU, mapping update, or debugger edit).
    void Invalidate(std::uint64_t addr, std::uint64_t size);

    [[nodiscard]] std::size_t SecondaryModuleCount() const;
    [[nodiscard]] std::size_t SecondaryFunctionCount() const;

private:
    struct RtBlock { // RtAPI first: a CPU.rt pointer converts back to the block
        RtAPI rt{};
        CodeManager* owner{};
    };

    struct InvalidatedRangeContext {
        std::vector<std::pair<std::uint64_t, std::uint64_t>>* ranges;
    };

    static void MissThunk(CPU* c, std::uint64_t pc);
    static void CodeChangedThunk(CPU* c, std::uint64_t addr);
    static int SkipThunk(void* context, std::uint64_t address);
    static void InvalidatedRangeThunk(void* context, std::uint64_t start, std::uint64_t end);

    void Miss(CPU* c, std::uint64_t pc);
    void CodeChanged(std::uint64_t addr);
    [[nodiscard]] bool IsInvalidated(std::uint64_t pc) const;
    void MergeInvalidatedRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges);
    [[nodiscard]] bool InPrimarySegment(std::uint64_t address) const;
    [[nodiscard]] GuestFn FindSecondary(std::uint64_t pc) const;
    GuestFn CompileSecondary(std::uint64_t pc, bool force = false);
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

    std::mutex m_compile_lock; // snx_recompiler is one-thread-at-a-time
    mutable std::shared_mutex m_secondary_lock;
    std::unordered_map<std::uint64_t, GuestFn> m_secondary;
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> m_invalidated_ranges;
    std::atomic<bool> m_has_invalidations{};
    std::atomic<bool> m_invalidation_active{};
    std::vector<snx_recompiled*> m_secondary_modules;
    std::function<void(std::size_t, std::size_t)> m_recompile_progress;
    std::atomic<std::uint64_t> m_primary_miss_count{};
    std::atomic<std::uint64_t> m_primary_miss_time_ns{};
    std::ofstream m_compiler_trace;
    std::vector<std::string> m_pending_compiler_log_lines;
};

} // namespace Core::Scarlet
