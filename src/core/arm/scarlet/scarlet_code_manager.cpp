// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/arm/scarlet/module_metadata.h"
#include "core/arm/scarlet/scarlet_code_manager.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef SCARLET_HARNESS
// Standalone harness (tools/scarlet-harness): print the format string only.
#define SCARLET_LOG(fmt, ...) std::fprintf(stderr, "%s\n", fmt)
#define SCARLET_LOG_INFO SCARLET_LOG
#define SCARLET_LOG_LINE(line) std::fprintf(stderr, "%s", line)
#define SCARLET_LOG_ERROR SCARLET_LOG
#define SCARLET_LOG_WARN SCARLET_LOG
#else
#include "common/logging.h"
#define SCARLET_LOG(...) LOG_DEBUG(Core_ARM, __VA_ARGS__)
#define SCARLET_LOG_INFO(...) LOG_INFO(Core_ARM, __VA_ARGS__)
#define SCARLET_LOG_ERROR(...) LOG_CRITICAL(Core_ARM, __VA_ARGS__)
#define SCARLET_LOG_WARN(...) LOG_WARNING(Core_ARM, __VA_ARGS__)
#endif

namespace Core::Scarlet {

namespace {
thread_local CodeManager* t_current_manager = nullptr;

// Instructions one just-in-time step compiles: the function reached and what it calls, breadth
// first (the figure ScarletNX's gnuabi runtime uses).
//
// Small on purpose. A step compiles everything breadth first from the missed pc whether or not
// the guest ever runs it, and each stale pc of a patched title is a step of its own: at 64 Ki
// instructions an UltraCam boot compiled 800 to 3000 functions per miss, 30 to 45 MiB of module
// span each, the same spans again and again. A few thousand instructions cover the function and
// its near callees; anything further is a miss of its own, compiled when it is reached.
constexpr std::size_t JitStepInstructions = 8192;
// Executable bytes read around a missed pc (half on each side). The module a step makes spans
// what it reached inside this window, and publishing walks that span word by word.
constexpr std::uint64_t MaxJitRange = 2ull * 1024 * 1024;

// Granularity of change detection for the primary module (IC IVAU's line size).
constexpr std::uint64_t LineSize = 64;
// Bytes read at a time when comparing a large range against the compiled snapshot.
constexpr std::uint64_t HashReadChunk = 1ull << 20;
// Attempts at publishing a secondary module while guest code keeps changing under it.
constexpr int MaxPublishAttempts = 8;

std::mutex g_stdout_capture_mutex;

std::uint64_t NextInstanceId() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

/// 64-bit hash of one line (or a shorter tail), a word at a time.
std::uint64_t HashLine(const std::uint8_t* data, std::size_t size) {
    std::uint64_t h = 0x9e3779b97f4a7c15ull ^ size;
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        std::uint64_t word;
        std::memcpy(&word, data + i, sizeof(word));
        h = (h ^ word) * 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    if (i < size) {
        std::uint64_t word = 0;
        std::memcpy(&word, data + i, size - i);
        h = (h ^ word) * 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 29;
    }
    return h;
}

/// Sorts and coalesces overlapping or touching [start, end) ranges in place.
void MergeRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
    std::sort(ranges.begin(), ranges.end());
    std::size_t out = 0;
    for (const auto& range : ranges) {
        if (out && ranges[out - 1].second >= range.first) {
            ranges[out - 1].second = std::max(ranges[out - 1].second, range.second);
        } else {
            ranges[out++] = range;
        }
    }
    ranges.resize(out);
}

class StdoutCapture {
public:
    using LineHandler = void (*)(void*, const std::string&);

    StdoutCapture(LineHandler handler, void* context)
        : m_lock{g_stdout_capture_mutex}, m_handler{handler}, m_context{context} {
        Start();
    }

    ~StdoutCapture() {
        Stop();
    }

    StdoutCapture(const StdoutCapture&) = delete;
    StdoutCapture& operator=(const StdoutCapture&) = delete;

    [[nodiscard]] bool Active() const {
        return m_redirected;
    }

private:
    void Start() {
#ifdef _WIN32
        HANDLE write_handle{};
        if (!CreatePipe(&m_read_handle, &write_handle, nullptr, 0))
            return;
        m_write_fd =
            _open_osfhandle(reinterpret_cast<std::intptr_t>(write_handle), _O_WRONLY | _O_BINARY);
        if (m_write_fd < 0) {
            CloseHandle(write_handle);
            CloseHandle(m_read_handle);
            m_read_handle = nullptr;
            return;
        }
        m_stdout_fd = _fileno(stdout);
        if (m_stdout_fd < 0)
            m_stdout_fd = 1;
        m_saved_fd = _dup(m_stdout_fd);
        if (m_saved_fd < 0 && _get_osfhandle(m_stdout_fd) != -1) {
            _close(m_write_fd);
            m_write_fd = -1;
            CloseHandle(m_read_handle);
            m_read_handle = nullptr;
            return;
        }
        m_had_stdout = m_saved_fd >= 0;
        m_previous_std_handle = GetStdHandle(STD_OUTPUT_HANDLE);
#else
        int pipe_fds[2];
        if (pipe(pipe_fds) != 0)
            return;
        for (int& fd : pipe_fds) {
            if (fd <= STDERR_FILENO) {
                const int replacement = fcntl(fd, F_DUPFD, STDERR_FILENO + 1);
                if (replacement < 0) {
                    close(pipe_fds[0]);
                    close(pipe_fds[1]);
                    return;
                }
                close(fd);
                fd = replacement;
            }
        }
        m_read_fd = pipe_fds[0];
        m_write_fd = pipe_fds[1];
        m_saved_fd = dup(STDOUT_FILENO);
        if (m_saved_fd < 0 && fcntl(STDOUT_FILENO, F_GETFD) != -1) {
            close(m_write_fd);
            close(m_read_fd);
            m_write_fd = m_read_fd = -1;
            return;
        }
        m_had_stdout = m_saved_fd >= 0;
#endif
        try {
            m_reader = std::thread{[this] { ReadPipe(); }};
        } catch (...) {
#ifdef _WIN32
            _close(m_write_fd);
            m_write_fd = -1;
#else
            close(m_write_fd);
            m_write_fd = -1;
#endif
            ClosePipe();
            if (m_saved_fd >= 0) {
#ifdef _WIN32
                _close(m_saved_fd);
#else
                close(m_saved_fd);
#endif
                m_saved_fd = -1;
            }
            return;
        }

        std::fflush(stdout);
#ifdef _WIN32
        if (_dup2(m_write_fd, m_stdout_fd) != 0) {
            _close(m_write_fd);
            m_write_fd = -1;
            m_reader.join();
            ClosePipe();
            if (m_saved_fd >= 0)
                _close(m_saved_fd);
            m_saved_fd = -1;
            return;
        }
        _close(m_write_fd);
        m_write_fd = -1;
        SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(m_stdout_fd)));
#else
        if (dup2(m_write_fd, STDOUT_FILENO) < 0) {
            close(m_write_fd);
            m_write_fd = -1;
            m_reader.join();
            ClosePipe();
            if (m_saved_fd >= 0)
                close(m_saved_fd);
            m_saved_fd = -1;
            return;
        }
        close(m_write_fd);
        m_write_fd = -1;
#endif
        m_redirected = true;
    }

    void Stop() {
        if (m_redirected) {
            std::fflush(stdout);
#ifdef _WIN32
            _dup2(m_saved_fd, m_stdout_fd);
            SetStdHandle(STD_OUTPUT_HANDLE, m_previous_std_handle);
            if (m_had_stdout)
                _close(m_saved_fd);
            else
                _close(m_stdout_fd);
#else
            if (m_had_stdout) {
                dup2(m_saved_fd, STDOUT_FILENO);
                close(m_saved_fd);
            } else {
                close(STDOUT_FILENO);
            }
#endif
            m_saved_fd = -1;
            m_redirected = false;
        }
        if (m_write_fd >= 0) {
#ifdef _WIN32
            _close(m_write_fd);
#else
            close(m_write_fd);
#endif
            m_write_fd = -1;
        }
        if (m_reader.joinable())
            m_reader.join();
        ClosePipe();
    }

    void ClosePipe() {
#ifdef _WIN32
        if (m_read_handle) {
            CloseHandle(m_read_handle);
            m_read_handle = nullptr;
        }
#else
        if (m_read_fd >= 0) {
            close(m_read_fd);
            m_read_fd = -1;
        }
#endif
    }

    void ReadPipe() {
        std::string pending;
        const auto consume = [this, &pending](const char* data, std::size_t size) {
            for (std::size_t i = 0; i < size; ++i) {
                if (data[i] == '\n') {
                    if (!pending.empty() && pending.back() == '\r')
                        pending.pop_back();
                    if (m_handler)
                        m_handler(m_context, pending);
                    pending.clear();
                } else {
                    pending.push_back(data[i]);
                }
            }
        };
        char buffer[4096];
#ifdef _WIN32
        DWORD bytes_read{};
        while (ReadFile(m_read_handle, buffer, sizeof(buffer), &bytes_read, nullptr) && bytes_read)
            consume(buffer, bytes_read);
#else
        ssize_t bytes_read{};
        while ((bytes_read = read(m_read_fd, buffer, sizeof(buffer))) > 0)
            consume(buffer, static_cast<std::size_t>(bytes_read));
#endif
        if (!pending.empty() && m_handler)
            m_handler(m_context, pending);
    }

    std::unique_lock<std::mutex> m_lock;
    LineHandler m_handler{};
    void* m_context{};
    std::thread m_reader;
    int m_saved_fd{-1};
    int m_write_fd{-1};
    bool m_had_stdout{};
    bool m_redirected{};
#ifdef _WIN32
    int m_stdout_fd{-1};
    HANDLE m_read_handle{};
    HANDLE m_previous_std_handle{};
#else
    int m_read_fd{-1};
#endif
};

std::filesystem::path PathFromUtf8(const char* path) {
    std::u8string encoded;
    while (*path) {
        encoded.push_back(static_cast<char8_t>(static_cast<unsigned char>(*path++)));
    }
    return std::filesystem::path{encoded};
}

std::string PathToUtf8(const std::filesystem::path& path) {
    const auto u8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

} // namespace

CodeManager::CodeManager(CodeSource& source)
    : m_source{source}, m_instance_id{NextInstanceId()} {
    m_block.owner = this;
}

CodeManager::~CodeManager() {
    if (!m_cache_dir.empty()) {
        SaveHints(m_cache_dir.c_str());
    }
    // Every CPU bound to the modules has stopped by now (the owner's contract).
    for (const auto& entry : m_secondary_modules) {
        snx_recompiled_free(entry.module);
    }
    if (m_primary) {
        snx_recompiled_free(m_primary);
    }
    if (m_jit) {
        snx_recompiler_free(m_jit);
    }
    if (m_recompiler) {
        snx_recompiler_free(m_recompiler);
    }
}

bool CodeManager::Initialize(const std::vector<Segment>& segments, const char* cache_dir_utf8,
                             std::string& error,
                             std::function<void(std::size_t, std::size_t)> progress,
                             const std::vector<ModuleImage>& modules) {
    bool cache_enabled = false;
    if (snx_recompiler_abi() != SNX_ABI) {
        error = "ScarletNX ABI " + std::to_string(snx_recompiler_abi()) + " differs from the " +
                std::to_string(SNX_ABI) + " this build was made against";
        return false;
    }
    m_recompiler = snx_recompiler_new(SNX_RECOMPILER_FLAG_LLVM);
    m_jit = snx_recompiler_new(SNX_RECOMPILER_FLAG_LLVM);
    if (!m_recompiler || !m_jit) {
        error = "the ScarletNX recompiler cannot target this host";
        return false;
    }
    snx_recompiler_set_optimization(m_recompiler, '2');
    snx_recompiler_set_optimization(m_jit, '1'); // compile latency matters mid-game
    snx_recompiler_set_hook_direct_calls(m_recompiler, 1);
    snx_recompiler_set_hook_direct_calls(m_jit, 1);
    snx_recompiler_set_verbose(m_recompiler, 1);
    snx_recompiler_set_verbose(m_jit, 1);
    std::string secondary_cache_utf8;
    if (cache_dir_utf8) {
        m_cache_dir = cache_dir_utf8;
        LoadHints(cache_dir_utf8);
        std::error_code ec;
        const auto cache_path = PathFromUtf8(cache_dir_utf8);
        std::filesystem::create_directories(cache_path, ec);
        const auto trace_path = cache_path / "scarletnx-recompiler.log";
        m_compiler_trace.open(trace_path, std::ios::out | std::ios::trunc);
        if (m_compiler_trace) {
            SCARLET_LOG_INFO("ScarletNX detailed compiler trace: {}", trace_path.string());
        } else {
            SCARLET_LOG_WARN("Could not open ScarletNX compiler trace at {}", trace_path.string());
        }

        const auto secondary_cache_path = cache_path / "secondary";
        std::filesystem::create_directories(secondary_cache_path, ec);
        secondary_cache_utf8 = PathToUtf8(secondary_cache_path);
    }
    m_recompile_progress = std::move(progress);
    if (cache_dir_utf8 && snx_recompiler_set_cache(m_recompiler, cache_dir_utf8) != 0) {
        SCARLET_LOG_WARN("ScarletNX module cache unavailable at {}", cache_dir_utf8);
    } else {
        cache_enabled = cache_dir_utf8 != nullptr;
        SCARLET_LOG_INFO("ScarletNX module cache directory: {}",
                         cache_dir_utf8 ? cache_dir_utf8 : "<disabled>");
    }
    if (!secondary_cache_utf8.empty() &&
        snx_recompiler_set_cache(m_jit, secondary_cache_utf8.c_str()) != 0) {
        SCARLET_LOG_WARN("ScarletNX secondary module cache unavailable at {}", secondary_cache_utf8);
    } else {
        m_secondary_cache_enabled = !secondary_cache_utf8.empty();
        if (m_secondary_cache_enabled) {
            SCARLET_LOG_INFO("ScarletNX secondary module cache directory: {}", secondary_cache_utf8);
        }
    }
    SCARLET_LOG_INFO(
        "ScarletNX compiler configuration: backend=LLVM primary=-O2 secondary=-O1 "
        "primary-cache={} secondary-cache={} detailed-compiler-log=scarletnx-recompiler.log",
        cache_enabled ? "enabled" : "disabled", m_secondary_cache_enabled ? "enabled" : "disabled");

    m_segments = segments;
    m_segment_hashes.clear();
    std::vector<std::vector<std::uint8_t>> blobs;
    std::vector<snx_code_range> ranges;
    std::vector<std::uint64_t> hints;
    std::uint64_t total_bytes = 0;
    blobs.reserve(segments.size());
    for (const auto& seg : segments) {
        auto& blob = blobs.emplace_back(seg.size);
        const auto read_started = std::chrono::steady_clock::now();
        m_source.Read(seg.addr, blob.data(), blob.size());
        const auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - read_started)
                                 .count();
        total_bytes += seg.size;
        auto& hashes = m_segment_hashes.emplace_back();
        hashes.reserve((seg.size + LineSize - 1) / LineSize);
        for (std::uint64_t off = 0; off < seg.size; off += LineSize) {
            hashes.push_back(HashLine(blob.data() + off,
                                      static_cast<std::size_t>(std::min(LineSize, seg.size - off))));
        }
        SCARLET_LOG("ScarletNX input snapshot: {:#x}-{:#x}, {} bytes, read {} ms", seg.addr,
                    seg.addr + seg.size, seg.size, static_cast<long long>(read_ms));
        ranges.push_back({seg.addr, blob.data(), blob.size()});
    }
    SCARLET_LOG("ScarletNX primary input: {} ranges, {} bytes", ranges.size(), total_bytes);
    // Keep these primary-module inputs stable across launches. ScarletNX's library cache stores
    // the entire linked module, whose key includes every discovered function and leader. Adding
    // runtime-discovered miss addresses to the seed list invalidates that whole-module cache on
    // each successive run. New addresses are compiled through the normal runtime miss path.
    for (const auto& module : modules) {
        if (!std::in_range<std::size_t>(module.image_size))
            continue;
        std::vector<std::uint8_t> image(static_cast<std::size_t>(module.image_size));
        m_source.Read(module.image_addr, image.data(), image.size());
        std::size_t unwind_entries{}, function_symbols{}, relocation_entries{};
        auto module_hints = DiscoverModuleEntryPoints(image, module.image_addr, module.text_offset,
                                                      module.text_size, unwind_entries,
                                                      function_symbols, relocation_entries);
        SCARLET_LOG("ScarletNX module metadata: image={:#x}+{} text={:#x}+{} entry points={} "
                    "(unwind={}, symbols={}, relocations={})",
                    module.image_addr, module.image_size, module.text_offset, module.text_size,
                    module_hints.size(), unwind_entries, function_symbols, relocation_entries);
        hints.insert(hints.end(), module_hints.begin(), module_hints.end());
    }
    std::sort(hints.begin(), hints.end());
    hints.erase(std::unique(hints.begin(), hints.end()), hints.end());
    SCARLET_LOG("ScarletNX AOT metadata: {} unique function, relocation, and unwind seeds",
                hints.size());
    const auto compile_started = std::chrono::steady_clock::now();
    m_pending_compiler_log_lines.clear();
    const auto output_line = [](void* context, const std::string& line) {
        static_cast<CodeManager*>(context)->HandleCompilerOutputLine(line);
    };
    bool captured_compiler_output{};
    {
        StdoutCapture capture{output_line, this};
        captured_compiler_output = capture.Active();
        if (!capture.Active())
            SCARLET_LOG_WARN(
                "Could not capture ScarletNX stdout; compiler output will go to the console");
        m_primary = snx_recompile_ranges(m_recompiler, ranges.data(), ranges.size(), hints.data(),
                                         hints.size());
    }
    const bool compiler_rebuilt_module = !m_pending_compiler_log_lines.empty();
    LogCapturedCompilerOutput();
    m_recompile_progress = {};
    const auto compile_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - compile_started)
                                .count();
    if (!m_primary) {
        error = snx_recompiler_error(m_recompiler);
        return false;
    }
    if (cache_enabled && captured_compiler_output) {
        SCARLET_LOG_INFO("ScarletNX primary module cache: {}", compiler_rebuilt_module
                                                                   ? "miss (compiled module)"
                                                                   : "hit (loaded module)");
    }
    SCARLET_LOG_INFO(
        "ScarletNX primary module ready in {} ms; compiler output is in scarletnx-recompiler.log",
        static_cast<long long>(compile_ms));

    snx_runtime_defaults(&m_block.rt); // ticks: 19.2 MHz monotonic; svc/undef/brk left NULL
    m_default_miss = m_block.rt.miss;
    m_block.rt.miss = &CodeManager::MissThunk;
    m_block.rt.code_changed = &CodeManager::CodeChangedThunk;
    m_block.rt.note_hint = &CodeManager::NoteHintThunk;
    WarmUpHints();
    // Keep the module's own miss path for its primary code. ScarletNX resolves and batches those
    // misses on its worker-backed JIT service; the manager handles only executable images outside
    // it.
    return true;
}

void CodeManager::HandleCompilerOutputLine(const std::string& line) {
    std::size_t done{}, total{};
    if (m_recompile_progress &&
        std::sscanf(line.c_str(), " compiled unit %zu/%zu", &done, &total) == 2)
        m_recompile_progress(done, total);
    if (m_compiler_trace) {
        m_compiler_trace << line << '\n';
        m_compiler_trace.flush();
    }
    // Keep the main emulator log concise, but surface the cache decision and phase totals
    // there so a test run can prove whether AOT actually reused a module.
    if (line.find("LLVM:") != std::string::npos)
        m_pending_compiler_log_lines.push_back(line);
}

void CodeManager::LogCapturedCompilerOutput() {
    for (const auto& line : m_pending_compiler_log_lines)
        SCARLET_LOG_INFO("ScarletNX: {}", line);
    m_pending_compiler_log_lines.clear();
}

bool CodeManager::Bind(CPU* cpu) {
    t_current_manager = this;
    cpu->rt = &m_block.rt;
    if (!m_primary || snx_recompiled_bind(m_primary, cpu) != 0) {
        return false;
    }
    bool validate_load = false;
    {
        std::scoped_lock lk{m_tables_lock};
        if (!m_load_staleness_checked) {
            m_load_staleness_checked = true;
            validate_load = true;
        }
    }
    if (validate_load) {
        // A module that came from the disk cache was compiled from the bytes its cache key names,
        // and the library only marks its own stale array when Invalidate tells it code changed:
        // guest bytes that already differ from those (a patch applied before this launch, or a
        // write the game makes while its code is being compiled) would otherwise run in the AOT
        // copy forever. Look once at load time, through the same path a later patch takes.
        for (const auto& seg : m_segments) {
            Invalidate(seg.addr, seg.size);
        }
    }
    return true;
}

void CodeManager::Run(CPU* cpu, std::uint64_t pc, std::uint64_t stop) {
    t_current_manager = this;
    if (!m_has_invalidations.load(std::memory_order_acquire)) {
        snx_recompiled_run(m_primary, cpu, pc, stop);
        return;
    }
    // Code that went stale diverts through the miss path and may return to this loop with cpu->pc
    // anywhere in guest code: a secondary module, or a retired primary function. Re-enter by pc.
    cpu->pc = pc;
    while (cpu->pc != stop) {
        const auto current = cpu->pc;
        if (!InPrimarySegment(current)) {
            if (!RunLiveFallback(cpu, current)) {
                cpu->rt->undef(cpu, current, 0);
            }
            continue;
        }
        // Retired primary code leaves its dispatch slots empty: the library's queued JIT
        // compiles what the guest has there now, from the copy Invalidate refreshed.
        snx_recompiled_call(m_primary, cpu, current);
    }
}

bool CodeManager::RunLiveFallback(CPU* cpu, std::uint64_t pc) {
    t_current_manager = this;
    GuestFn f = FindSecondary(pc);
    if (!f) {
        f = CompileSecondary(pc);
    }
    if (!f) {
        return false;
    }
    // Stale code can be entered very often; do not let the log become the cost.
    if (m_live_fallback_logs.fetch_add(1, std::memory_order_relaxed) < 16) {
        SCARLET_LOG_WARN("ScarletNX: stale code at {:#x}; running a live JIT copy", pc);
    }
    f(cpu, pc);
    return true;
}

bool CodeManager::InPrimarySegment(std::uint64_t address) const {
    for (const auto& seg : m_segments) {
        if (address >= seg.addr && address - seg.addr < seg.size) {
            return true;
        }
    }
    return false;
}

GuestFn CodeManager::FindSecondary(std::uint64_t pc) const {
    std::shared_lock lock{m_secondary_lock};
    const auto it = m_secondary.find(pc);
    return it == m_secondary.end() ? nullptr : it->second;
}

void CodeManager::NoteHintThunk(std::uint64_t pc) {
    if (auto* mgr = t_current_manager) {
        mgr->NoteHint(pc);
    }
}

void CodeManager::NoteHint(std::uint64_t pc) {
    if (!InPrimarySegment(pc) || (pc & 3)) {
        return;
    }
    std::scoped_lock lk{m_hints_lock};
    if (m_hint_set.insert(pc).second) {
        m_hints.push_back(pc);
        m_hints_dirty.store(true, std::memory_order_relaxed);
    }
}

std::vector<std::uint64_t> CodeManager::CopyHints() const {
    std::scoped_lock lk{m_hints_lock};
    return m_hints;
}

void CodeManager::SaveHints(const char* cache_dir_utf8) const {
    if (!cache_dir_utf8) {
        return;
    }
    const auto hints = CopyHints();
    if (hints.empty()) {
        return;
    }
    const auto dir = PathFromUtf8(cache_dir_utf8);
    const auto path = dir / "runtime.hints";
    const auto tmp = dir / "runtime.hints.tmp";
    std::ofstream out(tmp, std::ios::out | std::ios::trunc);
    if (!out) {
        return;
    }
    for (const auto pc : hints) {
        out << std::hex << pc << '\n';
    }
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    SCARLET_LOG_INFO("ScarletNX: saved {} runtime hints to {}", hints.size(), path.string());
}

void CodeManager::PeriodicSaveHints() {
    if (!m_cache_dir.empty() && m_hints_dirty.exchange(false, std::memory_order_relaxed)) {
        SaveHints(m_cache_dir.c_str());
    }
}

void CodeManager::LoadHints(const char* cache_dir_utf8) {
    if (!cache_dir_utf8) {
        return;
    }
    const auto dir = PathFromUtf8(cache_dir_utf8);
    const auto path = dir / "runtime.hints";
    std::size_t loaded = 0;
    std::ifstream in(path);
    if (in) {
        std::string line;
        std::scoped_lock lk{m_hints_lock};
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto pc = std::strtoull(line.c_str(), nullptr, 16);
            if (pc && !(pc & 3) && m_hint_set.insert(pc).second) {
                m_hints.push_back(pc);
                loaded++;
            }
        }
    }

    // Also recover any precompiled unit addresses already in library-jit.cache
    const auto jit_cache_path = dir / "library-jit.cache";
    std::ifstream jit_in(jit_cache_path, std::ios::binary | std::ios::ate);
    if (jit_in) {
        const auto file_size = static_cast<std::size_t>(jit_in.tellg());
        if (file_size > 12) {
            std::vector<std::uint8_t> buffer(file_size);
            jit_in.seekg(0, std::ios::beg);
            if (jit_in.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
                constexpr std::uint32_t RECORD_MAGIC = 0x524A4E53u;
                if (std::memcmp(buffer.data(), "SNXJIT01", 8) == 0) {
                    std::uint32_t key_len = 0;
                    std::memcpy(&key_len, buffer.data() + 8, sizeof(key_len));
                    std::size_t pos = 12 + key_len;
                    std::scoped_lock lk{m_hints_lock};
                    while (pos + 12 <= file_size) {
                        std::uint32_t rmagic = 0;
                        std::uint64_t rlength = 0;
                        std::memcpy(&rmagic, buffer.data() + pos, sizeof(rmagic));
                        std::memcpy(&rlength, buffer.data() + pos + 4, sizeof(rlength));
                        if (rmagic != RECORD_MAGIC || rlength > file_size - pos - 12) {
                            break;
                        }
                        const std::size_t body_pos = pos + 12;
                        if (body_pos + 2 <= file_size) {
                            std::uint16_t name_len = 0;
                            std::memcpy(&name_len, buffer.data() + body_pos, sizeof(name_len));
                            std::size_t p = body_pos + 2 + name_len;
                            if (p + 4 <= file_size) {
                                std::uint32_t fn_count = 0;
                                std::memcpy(&fn_count, buffer.data() + p, sizeof(fn_count));
                                p += 4;
                                for (std::uint32_t f = 0; f < fn_count && p + 12 <= file_size; ++f) {
                                    std::uint64_t start = 0;
                                    std::memcpy(&start, buffer.data() + p, sizeof(start));
                                    if (start && !(start & 3) && m_hint_set.insert(start).second) {
                                        m_hints.push_back(start);
                                        loaded++;
                                    }
                                    std::uint32_t leaders = 0;
                                    std::memcpy(&leaders, buffer.data() + p + 8, sizeof(leaders));
                                    p += 12 + 8 * leaders;
                                    if (p + 4 > file_size) break;
                                    std::uint32_t runs = 0;
                                    std::memcpy(&runs, buffer.data() + p, sizeof(runs));
                                    p += 4 + 12 * runs + 16;
                                }
                            }
                        }
                        pos = body_pos + rlength + 8;
                    }
                }
            }
        }
    }

    if (loaded) {
        SCARLET_LOG_INFO("ScarletNX: loaded {} runtime hints from cache", loaded);
        SaveHints(cache_dir_utf8);
    }
}

void CodeManager::WarmUpHints() {
    if (!m_primary) {
        return;
    }
    const auto hints = CopyHints();
    std::size_t warmed = 0;
    for (const auto pc : hints) {
        if (InPrimarySegment(pc) && !snx_recompiled_function(m_primary, pc)) {
            if (snx_recompiled_lookup(m_primary, pc)) {
                warmed++;
            }
        }
    }
    if (warmed) {
        SCARLET_LOG_INFO("ScarletNX: warm-started {} JIT units from hint cache", warmed);
    }
}

void CodeManager::MissThunk(CPU* c, std::uint64_t pc) {
    reinterpret_cast<const RtBlock*>(c->rt)->owner->Miss(c, pc);
}

void CodeManager::CodeChangedThunk(CPU* c, std::uint64_t addr) {
    reinterpret_cast<const RtBlock*>(c->rt)->owner->CodeChanged(addr);
}

void CodeManager::CodeChanged(std::uint64_t addr) {
    const auto start = addr & ~(LineSize - 1);
    Invalidate(start, LineSize);
}

// Which already-compiled code a secondary compile need not redo. A primary address counts only
// while its primary function is still valid, or when a live copy of it exists.
int CodeManager::SkipThunk(void* context, std::uint64_t address) {
    auto* self = static_cast<CodeManager*>(context);
    if (self->FindSecondary(address) != nullptr) {
        return 1;
    }
    if (self->InPrimarySegment(address)) {
        // The primary module owns these, stale or not: a call into one goes through its dispatch
        // table and the library's JIT, which compiles the current code.
        return 1;
    }
    return 0;
}

void CodeManager::Miss(CPU* c, std::uint64_t pc) {
    if (pc & 3) {
        c->rt->undef(c, pc, 0);
        return;
    }
    if (InPrimarySegment(pc)) {
        NoteHint(pc);
        const auto started = std::chrono::steady_clock::now();
        if (!m_default_miss) {
            // No queued JIT to ask (the library gave no default miss): the live copy is all
            // there is.
            if (!RunLiveFallback(c, pc))
                c->rt->undef(c, pc, 0);
        } else if (m_default_miss) {
            // A cold pc of the primary module: the library's RtAPI.miss queues it on ScarletNX's
            // batched, cache-first JIT worker pool, which re-arms the slot and resumes there.
            // This function's contract is stronger than that: MissThunk is the module's fallback
            // ("compiles the code there and runs it"), so pc must have run before this returns.
            // The queued JIT can answer without running it -- its pending unit for the pc was
            // already consumed (jit_service.c:78: bit set, nothing queued, so a miss gets NULL
            // without compiling), or the pc is not inside the frozen images it compiles from.
            // Returning on that path unwinds one host frame with cpu->pc still at pc: the guest
            // walks back out of its own call stack, re-missing at every outer frame (a run of
            // this title logged 1.24 million misses, each one a fraction of a millisecond), and
            // ends in a thread whose registers hold string bytes and a read of unmapped memory.
            // Require progress instead: cpu->pc moved off pc, otherwise run the live copy.
            m_default_miss(c, pc);
            if (c->pc == pc) {
                const auto taken =
                    m_miss_no_progress_fallbacks.fetch_add(1, std::memory_order_relaxed) + 1;
                if (taken <= 16) {
                    SCARLET_LOG_WARN(
                        "ScarletNX: queued JIT took {:#x} and ran nothing; running a live JIT copy",
                        pc);
                }
                if (!RunLiveFallback(c, pc)) {
                    c->rt->undef(c, pc, 0);
                }
            }
        } else {
            c->rt->undef(c, pc, 0);
        }
        const auto elapsed_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - started)
                                           .count());
        const auto miss = m_primary_miss_count.fetch_add(1, std::memory_order_relaxed) + 1;
        m_primary_miss_time_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
        if (miss <= 32 || miss % 1000 == 0) {
            SCARLET_LOG(
                "ScarletNX primary queued JIT miss #{} at {:#x}: wait {} ms; cumulative {} ms",
                miss, pc, elapsed_ns / 1000000,
                m_primary_miss_time_ns.load(std::memory_order_relaxed) / 1000000);
        }
        return;
    }
    GuestFn f = FindSecondary(pc);
    if (!f) {
        f = CompileSecondary(pc);
    }
    if (!f) {
        c->rt->undef(c, pc, 0); // not executable, or not compilable: the owner's abort path
        return;
    }
    f(c, pc);
}

GuestFn CodeManager::CompileSecondary(std::uint64_t pc) {
    std::scoped_lock compile{m_compile_lock};
    for (int attempt = 0; attempt < MaxPublishAttempts; ++attempt) {
        if (GuestFn f = FindSecondary(pc)) {
            return f; // another core compiled it while this one waited
        }
        // Read before the guest bytes: an Invalidate that starts after this makes the publish
        // below re-check the source against memory.
        const auto generation = m_invalidate_gen.load(std::memory_order_acquire);

        std::uint64_t lo = 0, hi = 0;
        if (!m_source.ExecRangeAround(pc, MaxJitRange, lo, hi) || pc < lo || pc >= hi) {
            SCARLET_LOG_ERROR("ScarletNX: call to non-executable address {:#x}", pc);
            return nullptr;
        }
        const auto read_started = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> bytes(hi - lo);
        m_source.Read(lo, bytes.data(), bytes.size());
        const auto read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - read_started)
                                 .count();
        const snx_code_range range{lo, bytes.data(), bytes.size()};

        const auto started = std::chrono::steady_clock::now();
        SCARLET_LOG("ScarletNX secondary JIT start: pc={:#x}, range={:#x}-{:#x} ({} bytes), "
                    "read={} ms",
                    pc, lo, hi, bytes.size(), static_cast<long long>(read_ms));
        const auto output_line = [](void* context, const std::string& line) {
            static_cast<CodeManager*>(context)->HandleCompilerOutputLine(line);
        };
        m_pending_compiler_log_lines.clear();
        snx_recompiled* module{};
        bool captured_compiler_output{};
        {
            StdoutCapture capture{output_line, this};
            captured_compiler_output = capture.Active();
            if (!capture.Active())
                SCARLET_LOG_WARN(
                    "Could not capture ScarletNX stdout during secondary JIT compilation");
            module = snx_recompile_from(m_jit, &range, 1, &pc, 1, JitStepInstructions,
                                        &CodeManager::SkipThunk, this);
        }
        const bool compiler_rebuilt_module = !m_pending_compiler_log_lines.empty();
        LogCapturedCompilerOutput();
        if (!module) {
            SCARLET_LOG_ERROR("ScarletNX: cannot compile code at {:#x}: {}", pc,
                              snx_recompiler_error(m_jit));
            return nullptr;
        }
        if (m_secondary_cache_enabled && captured_compiler_output) {
            SCARLET_LOG_INFO("ScarletNX secondary module cache at {:#x}: {}", pc,
                             compiler_rebuilt_module ? "miss (compiled module)"
                                                     : "hit (loaded module)");
        }
        std::uint64_t module_lo = 0, module_hi = 0;
        snx_recompiled_range(module, &module_lo, &module_hi);

        std::size_t functions = 0;
        bool changed_underneath = false;
        // The entry for pc, taken from the module itself: an Invalidate may evict it from the
        // table the moment the lock below drops, and that must not turn into an undefined
        // instruction. Code that went stale diverts through the miss path on its own.
        const GuestFn entry = snx_recompiled_function(module, pc);
        {
            std::unique_lock lock{m_secondary_lock};
            if (m_invalidate_gen.load(std::memory_order_acquire) != generation &&
                module_lo >= lo && module_hi <= hi && module_lo < module_hi) {
                // Guest code was invalidated while this compiled. Invalidate evicts under this
                // same lock, so a module published now is either evicted after, or was
                // compiled from bytes memory still holds. Check which.
                std::vector<std::uint8_t> current(module_hi - module_lo);
                m_source.Read(module_lo, current.data(), current.size());
                changed_underneath =
                    std::memcmp(current.data(), bytes.data() + (module_lo - lo), current.size()) !=
                    0;
            }
            if (!changed_underneath) {
                for (std::uint64_t a = module_lo; a < module_hi; a += 4) {
                    if (GuestFn f = snx_recompiled_function(module, a)) {
                        m_secondary[a] = f;
                        functions++;
                    }
                }
                // Record the bytes this module describes, one hash per line: an invalidation
                // whose range turns out to hold the same bytes (the whole-cache clearing SVC, an
                // undefined-instruction window that was never written) must not retire it, or
                // every address inside it compiles a fresh module again and the address space
                // runs out.
                std::vector<std::uint64_t> line_hashes;
                line_hashes.reserve((module_hi - module_lo) / LineSize + 1);
                for (std::size_t off = 0; module_lo + off < module_hi; off += LineSize) {
                    const auto length =
                        std::min<std::uint64_t>(LineSize, module_hi - (module_lo + off));
                    line_hashes.push_back(HashLine(bytes.data() + (module_lo - lo) + off,
                                                   static_cast<std::size_t>(length)));
                }
                m_secondary_modules.push_back(
                    {module, module_lo, module_hi, std::move(line_hashes)});
            }
        }
        if (changed_underneath) {
            // Never published, so nothing can be running it.
            snx_recompiled_free(module);
            SCARLET_LOG("ScarletNX: code at {:#x}-{:#x} changed during compilation; retrying",
                        module_lo, module_hi);
            continue;
        }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started)
                            .count();
        SCARLET_LOG("ScarletNX JIT {:#x}: {} functions in {:#x}-{:#x}, {} ms", pc, functions,
                    module_lo, module_hi, static_cast<long long>(ms));
        return entry;
    }
    SCARLET_LOG_ERROR("ScarletNX: code at {:#x} kept changing while it was compiled", pc);
    return nullptr;
}

void CodeManager::InvalidatedRangeThunk(void* context, std::uint64_t start, std::uint64_t end) {
    auto* ranges = static_cast<InvalidatedRangeContext*>(context)->ranges;
    if (start < end)
        ranges->emplace_back(start, end);
}

CodeManager::RangeList CodeManager::ChangedPrimaryRuns(std::uint64_t addr,
                                                       std::uint64_t end) const {
    RangeList runs;
    std::vector<std::uint8_t> buffer;
    for (std::size_t i = 0; i < m_segments.size(); ++i) {
        const auto& seg = m_segments[i];
        const auto seg_end = seg.addr + seg.size;
        if (addr >= seg_end || seg.addr >= end)
            continue;
        const auto lo = std::max(addr, seg.addr);
        const auto hi = std::min(end, seg_end);
        if (lo >= hi)
            continue;
        const auto& hashes = m_segment_hashes[i];
        // Whole lines, from the line holding lo through the line holding hi - 1.
        const auto first_line = (lo - seg.addr) / LineSize;
        const auto last_line = (hi - 1 - seg.addr) / LineSize;
        for (std::uint64_t chunk = first_line; chunk <= last_line;) {
            const auto chunk_lines = std::min<std::uint64_t>(last_line - chunk + 1,
                                                             HashReadChunk / LineSize);
            const auto chunk_addr = seg.addr + chunk * LineSize;
            const auto chunk_end = std::min(seg_end, chunk_addr + chunk_lines * LineSize);
            buffer.resize(chunk_end - chunk_addr);
            m_source.Read(chunk_addr, buffer.data(), buffer.size());
            for (std::uint64_t line = 0; line < chunk_lines; ++line) {
                const auto offset = line * LineSize;
                const auto length = std::min<std::uint64_t>(LineSize, buffer.size() - offset);
                if (HashLine(buffer.data() + offset, static_cast<std::size_t>(length)) ==
                    hashes[chunk + line]) {
                    continue;
                }
                const auto start = chunk_addr + offset;
                if (!runs.empty() && runs.back().second == start)
                    runs.back().second = start + length;
                else
                    runs.emplace_back(start, start + length);
            }
            chunk += chunk_lines;
        }
    }
    return runs;
}

// Does a published just-in-time module still describe the memory in [addr, end)? Same walk as the
// primary comparison above, over one module's stored line hashes, and stopped at the first line
// that differs.
bool CodeManager::SecondarySourceChanged(const SecondaryModule& entry, std::uint64_t addr,
                                         std::uint64_t end) const {
    const auto lo = std::max(addr, entry.lo);
    const auto hi = std::min(end, entry.hi);
    if (lo >= hi) {
        return false;
    }
    std::vector<std::uint8_t> buffer;
    const auto first_line = (lo - entry.lo) / LineSize;
    const auto last_line = (hi - 1 - entry.lo) / LineSize;
    for (std::uint64_t chunk = first_line; chunk <= last_line;) {
        const auto chunk_lines =
            std::min<std::uint64_t>(last_line - chunk + 1, HashReadChunk / LineSize);
        const auto chunk_addr = entry.lo + chunk * LineSize;
        const auto chunk_end = std::min(entry.hi, chunk_addr + chunk_lines * LineSize);
        buffer.resize(chunk_end - chunk_addr);
        m_source.Read(chunk_addr, buffer.data(), buffer.size());
        for (std::uint64_t line = 0; line < chunk_lines; ++line) {
            const auto offset = line * LineSize;
            const auto length = std::min<std::uint64_t>(LineSize, buffer.size() - offset);
            if (chunk + line < entry.line_hashes.size() &&
                HashLine(buffer.data() + offset, static_cast<std::size_t>(length)) ==
                    entry.line_hashes[chunk + line]) {
                continue;
            }
            return true;
        }
        chunk += chunk_lines;
    }
    return false;
}

// The stored hashes follow the bytes memory holds now, so the second report of one patch (the
// page table's unmapping and the guest's own IC IVAU both describe the same write) retires and
// logs nothing again, and a later write of different bytes is found again. Same walk as the
// comparison above, storing instead of comparing.
void CodeManager::RefreshPrimaryHashes(std::uint64_t addr, std::uint64_t end) {
    std::vector<std::uint8_t> buffer;
    for (std::size_t i = 0; i < m_segments.size(); ++i) {
        const auto& seg = m_segments[i];
        const auto seg_end = seg.addr + seg.size;
        if (addr >= seg_end || seg.addr >= end || i >= m_segment_hashes.size()) {
            continue;
        }
        const auto lo = std::max(addr, seg.addr);
        const auto hi = std::min(end, seg_end);
        if (lo >= hi) {
            continue;
        }
        auto& hashes = m_segment_hashes[i];
        const auto first_line = (lo - seg.addr) / LineSize;
        const auto last_line = (hi - 1 - seg.addr) / LineSize;
        for (std::uint64_t chunk = first_line; chunk <= last_line;) {
            const auto chunk_lines =
                std::min<std::uint64_t>(last_line - chunk + 1, HashReadChunk / LineSize);
            const auto chunk_addr = seg.addr + chunk * LineSize;
            const auto chunk_end = std::min(seg_end, chunk_addr + chunk_lines * LineSize);
            buffer.resize(chunk_end - chunk_addr);
            m_source.Read(chunk_addr, buffer.data(), buffer.size());
            for (std::uint64_t line = 0; line < chunk_lines; ++line) {
                const auto offset = line * LineSize;
                const auto length = std::min<std::uint64_t>(LineSize, buffer.size() - offset);
                if (chunk + line < hashes.size()) {
                    hashes[chunk + line] =
                        HashLine(buffer.data() + offset, static_cast<std::size_t>(length));
                }
            }
            chunk += chunk_lines;
        }
    }
}

void CodeManager::Invalidate(std::uint64_t addr, std::uint64_t size) {
    if (size > UINT64_MAX - addr)
        size = UINT64_MAX - addr;
    if (!size)
        return;
    const std::uint64_t end = addr + size;
    std::scoped_lock serialize{m_invalidate_lock};
    m_invalidate_gen.fetch_add(1, std::memory_order_acq_rel);

    // Primary: only lines whose bytes differ from what the module was compiled from. The changed
    // lines are published first so a miss in them goes to live code even before the library has
    // retired the functions covering them.
    if (m_primary) {
        auto changed = ChangedPrimaryRuns(addr, end);
        if (!changed.empty()) {
            RangeList stale = changed;
            // The library's JIT compiles from its own copy of the code, frozen when the module
            // was made; bring that copy up to what the guest has now, before the functions are
            // retired, so the first miss after that asks the JIT for the new code (the NX
            // runtime's JIT reads the guest's memory itself).
            std::size_t refilled = 0;
            std::vector<std::uint8_t> now;
            for (const auto& [start, stop] : changed) {
                now.resize(stop - start);
                m_source.Read(start, now.data(), now.size());
                refilled += snx_recompiled_update_code(m_primary, start, now.data(), now.size());
            }
            std::size_t retired = 0;
            InvalidatedRangeContext context{&stale};
            for (const auto& [start, stop] : changed) {
                retired += snx_recompiled_invalidate(m_primary, start, stop - start,
                                                     &InvalidatedRangeThunk, &context);
            }
            std::uint64_t span_lo = addr;
            std::uint64_t span_hi = end;
            for (const auto& [start, stop] : stale) {
                span_lo = std::min(span_lo, start);
                span_hi = std::max(span_hi, stop);
            }
            RefreshPrimaryHashes(addr, end);
            if (retired || refilled) {
                SCARLET_LOG_INFO("ScarletNX invalidated {} primary AOT function(s) and refreshed {} code byte(s) for {:#x}-{:#x}",
                                 retired, refilled, span_lo, span_hi);
            }
        }
    }

    // Secondary: the modules overlapping the range, then every cached leader in their retired
    // spans and in the range itself. All under the publish lock, so a compile that finishes
    // concurrently is either evicted here or re-verified against memory by its own publish.
    // Retired native code stays loaded: another guest thread may still be executing it.
    RangeList evict;
    evict.emplace_back(addr, end);
    InvalidatedRangeContext evict_context{&evict};
    std::unique_lock lock{m_secondary_lock};
    std::size_t kept = 0;
    for (const auto& entry : m_secondary_modules) {
        if (entry.hi <= addr || entry.lo >= end) {
            continue;
        }
        // A module whose lines under this range still hold the bytes it was compiled from says
        // nothing about this invalidation: keeping it is what stops a whole-cache clearing SVC,
        // or an undefined-instruction window, from retiring every just-in-time module it meets.
        if (!entry.line_hashes.empty() && !SecondarySourceChanged(entry, addr, end)) {
            kept++;
            continue;
        }
        snx_recompiled_invalidate(entry.module, addr, size, &InvalidatedRangeThunk, &evict_context);
        // Running code of this module may now divert and return to Run() mid-function.
        m_has_invalidations.store(true, std::memory_order_release);
    }
    if (kept) {
        SCARLET_LOG("ScarletNX kept {} just-in-time module(s) over {:#x}-{:#x}; the bytes there did "
                    "not change",
                    kept, addr, end);
    }
    MergeRanges(evict);
    std::uint64_t probes = 0;
    for (const auto& [start, stop] : evict) {
        const auto span = stop - start; // no "+ 3": it wraps for the whole address space
        const auto count = span / 4 + ((span & 3) ? 1 : 0);
        probes = count > UINT64_MAX - probes ? UINT64_MAX : probes + count;
    }
    if (probes <= m_secondary.size()) {
        // Small change (an IC IVAU line): look the entries up instead of scanning the table.
        for (const auto& [start, stop] : evict) {
            for (std::uint64_t pc = (start + 3) & ~std::uint64_t{3}; pc < stop; pc += 4) {
                m_secondary.erase(pc);
                if (pc > UINT64_MAX - 4)
                    break;
            }
        }
    } else {
        for (auto it = m_secondary.begin(); it != m_secondary.end();) {
            const auto range = std::upper_bound(
                evict.begin(), evict.end(), it->first,
                [](std::uint64_t address, const auto& candidate) { return address < candidate.first; });
            const bool stale = range != evict.begin() && it->first < std::prev(range)->second;
            if (stale) {
                it = m_secondary.erase(it);
            } else {
                it = std::next(it);
            }
        }
    }
}

std::size_t CodeManager::SecondaryModuleCount() const {
    std::shared_lock lock{m_secondary_lock};
    return m_secondary_modules.size();
}

std::size_t CodeManager::SecondaryFunctionCount() const {
    std::shared_lock lock{m_secondary_lock};
    return m_secondary.size();
}

} // namespace Core::Scarlet
