// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/arm/scarlet/module_metadata.h"
#include "core/arm/scarlet/scarlet_code_manager.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
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
// Instructions one just-in-time step compiles: the function reached and what it calls, breadth
// first (the figure ScarletNX's gnuabi runtime uses).
constexpr std::size_t JitStepInstructions = 65536;
// Executable bytes read around a missed pc. NRO text is a few MiB; the library's tables cover at
// most 1 GiB.
constexpr std::uint64_t MaxJitRange = 64ull * 1024 * 1024;

std::mutex g_stdout_capture_mutex;

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

} // namespace

CodeManager::CodeManager(CodeSource& source) : m_source{source} {
    m_block.owner = this;
}

CodeManager::~CodeManager() {
    // Every CPU bound to the modules has stopped by now (the owner's contract).
    for (auto* module : m_secondary_modules) {
        snx_recompiled_free(module);
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
    if (cache_dir_utf8) {
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
    }
    m_recompile_progress = std::move(progress);
    if (cache_dir_utf8 && snx_recompiler_set_cache(m_recompiler, cache_dir_utf8) != 0) {
        SCARLET_LOG_WARN("ScarletNX module cache unavailable at {}", cache_dir_utf8);
    } else {
        cache_enabled = cache_dir_utf8 != nullptr;
        SCARLET_LOG_INFO("ScarletNX module cache directory: {}",
                         cache_dir_utf8 ? cache_dir_utf8 : "<disabled>");
    }
    if (cache_dir_utf8 && snx_recompiler_set_cache(m_jit, cache_dir_utf8) != 0) {
        SCARLET_LOG_WARN("ScarletNX secondary module cache unavailable at {}", cache_dir_utf8);
    } else {
        m_secondary_cache_enabled = cache_dir_utf8 != nullptr;
    }
    SCARLET_LOG_INFO(
        "ScarletNX compiler configuration: backend=LLVM primary=-O2 secondary=-O1 "
        "primary-cache={} secondary-cache={} detailed-compiler-log=scarletnx-recompiler.log",
        cache_enabled ? "enabled" : "disabled", m_secondary_cache_enabled ? "enabled" : "disabled");

    m_segments = segments;
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
    cpu->rt = &m_block.rt;
    return m_primary && snx_recompiled_bind(m_primary, cpu) == 0;
}

void CodeManager::Run(CPU* cpu, std::uint64_t pc, std::uint64_t stop) {
    if (!m_has_invalidations.load(std::memory_order_acquire) &&
        !m_invalidation_active.load(std::memory_order_acquire)) {
        snx_recompiled_run(m_primary, cpu, pc, stop);
        return;
    }
    cpu->pc = pc;
    while (cpu->pc != stop) {
        const auto current = cpu->pc;
        if (m_invalidation_active.load(std::memory_order_acquire)) {
            [[maybe_unused]] std::scoped_lock barrier{m_compile_lock};
        }
        if (IsInvalidated(current)) {
            if (!RunLiveFallback(cpu, current)) {
                cpu->rt->undef(cpu, current, 0);
            }
        } else {
            snx_recompiled_call(m_primary, cpu, current);
        }
    }
}

bool CodeManager::RunLiveFallback(CPU* cpu, std::uint64_t pc) {
    GuestFn f = CompileSecondary(pc, true);
    if (!f) {
        return false;
    }
    SCARLET_LOG_WARN("ScarletNX: stale code at {:#x}; running a live JIT copy", pc);
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

void CodeManager::MissThunk(CPU* c, std::uint64_t pc) {
    reinterpret_cast<const RtBlock*>(c->rt)->owner->Miss(c, pc);
}

void CodeManager::CodeChangedThunk(CPU* c, std::uint64_t addr) {
    reinterpret_cast<const RtBlock*>(c->rt)->owner->CodeChanged(addr);
}

void CodeManager::CodeChanged(std::uint64_t addr) {
    constexpr std::uint64_t cache_line_size = 64;
    const auto start = addr & ~(cache_line_size - 1);
    Invalidate(start, cache_line_size);
}

int CodeManager::SkipThunk(void* context, std::uint64_t address) {
    auto* self = static_cast<CodeManager*>(context);
    if (self->InPrimarySegment(address)) {
        return snx_recompiled_function(self->m_primary, address) != nullptr;
    }
    return self->FindSecondary(address) != nullptr;
}

void CodeManager::Miss(CPU* c, std::uint64_t pc) {
    if (pc & 3) {
        c->rt->undef(c, pc, 0);
        return;
    }
    const bool primary = InPrimarySegment(pc);
    const auto started = std::chrono::steady_clock::now();
    if (m_invalidation_active.load(std::memory_order_acquire)) {
        [[maybe_unused]] std::scoped_lock barrier{m_compile_lock};
    }
    if (primary) {
        if (IsInvalidated(pc)) {
            if (!RunLiveFallback(c, pc))
                c->rt->undef(c, pc, 0);
            return;
        }
        // The library's RtAPI.miss waits for ScarletNX's batched JIT worker pool. Do not create
        // a synchronous one-function module for every primary dispatch miss.
        if (m_default_miss) {
            m_default_miss(c, pc);
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

GuestFn CodeManager::CompileSecondary(std::uint64_t pc, bool force) {
    std::scoped_lock compile{m_compile_lock};
    if (GuestFn f = FindSecondary(pc)) {
        return f; // another core compiled it while this one waited
    }

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
    SCARLET_LOG("ScarletNX secondary JIT start: pc={:#x}, range={:#x}-{:#x} ({} bytes), forced={}, "
                "read={} ms",
                pc, lo, hi, bytes.size(), force, static_cast<long long>(read_ms));
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
            SCARLET_LOG_WARN("Could not capture ScarletNX stdout during secondary JIT compilation");
        module =
            snx_recompile_from(m_jit, &range, 1, &pc, 1, JitStepInstructions,
                               force ? nullptr : &CodeManager::SkipThunk, force ? nullptr : this);
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
    {
        std::unique_lock lock{m_secondary_lock};
        for (std::uint64_t a = module_lo; a < module_hi; a += 4) {
            if (GuestFn f = snx_recompiled_function(module, a)) {
                m_secondary[a] = f;
                functions++;
            }
        }
        m_secondary_modules.push_back(module);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count();
    SCARLET_LOG("ScarletNX JIT {:#x}: {} functions in {:#x}-{:#x}, {} ms", pc, functions, module_lo,
                module_hi, static_cast<long long>(ms));
    return FindSecondary(pc);
}

void CodeManager::InvalidatedRangeThunk(void* context, std::uint64_t start, std::uint64_t end) {
    auto* ranges = static_cast<InvalidatedRangeContext*>(context)->ranges;
    if (start < end)
        ranges->emplace_back(start, end);
}

void CodeManager::MergeInvalidatedRanges(
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (ranges.empty())
        return;
    const auto previous = std::atomic_load_explicit(&m_invalidated_ranges, std::memory_order_relaxed);
    if (previous)
        ranges.insert(ranges.end(), previous->begin(), previous->end());
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    merged.reserve(ranges.size());
    for (const auto& range : ranges) {
        if (merged.empty() || merged.back().second < range.first) {
            merged.push_back(range);
        } else {
            merged.back().second = std::max(merged.back().second, range.second);
        }
    }
    std::atomic_store_explicit(
        &m_invalidated_ranges,
        std::make_shared<const std::vector<std::pair<std::uint64_t, std::uint64_t>>>(
            std::move(merged)),
        std::memory_order_release);
    m_has_invalidations.store(true, std::memory_order_release);
}

bool CodeManager::IsInvalidated(std::uint64_t pc) const {
    const auto ranges = std::atomic_load_explicit(&m_invalidated_ranges, std::memory_order_acquire);
    if (!ranges)
        return false;
    const auto it = std::upper_bound(
        ranges->begin(), ranges->end(), pc,
        [](std::uint64_t address, const auto& range) { return address < range.first; });
    return it != ranges->begin() && pc < std::prev(it)->second;
}

void CodeManager::Invalidate(std::uint64_t addr, std::uint64_t size) {
    if (size > UINT64_MAX - addr)
        size = UINT64_MAX - addr;
    if (!size)
        return;
    std::scoped_lock compile{m_compile_lock};
    m_invalidation_active.store(true, std::memory_order_release);
    struct ActiveReset {
        std::atomic<bool>& active;
        ~ActiveReset() {
            active.store(false, std::memory_order_release);
        }
    } active_reset{m_invalidation_active};
    const std::uint64_t end = addr + size;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> invalidated_ranges;
    InvalidatedRangeContext context{&invalidated_ranges};
    if (addr < end)
        invalidated_ranges.emplace_back(addr, end);
    if (m_primary) {
        const auto invalidated =
            snx_recompiled_invalidate(m_primary, addr, size, &InvalidatedRangeThunk, &context);
        if (invalidated) {
            SCARLET_LOG_INFO("ScarletNX invalidated {} primary AOT function(s) for {:#x}-{:#x}",
                             invalidated, addr, end);
        }
        for (const auto& segment : m_segments) {
            const auto segment_end = segment.addr + segment.size;
            if (addr < segment_end && segment.addr < end) {
                invalidated_ranges.emplace_back(std::max(addr, segment.addr),
                                                std::min(end, segment_end));
            }
        }
    }
    std::vector<snx_recompiled*> modules;
    {
        std::shared_lock lock{m_secondary_lock};
        modules = m_secondary_modules;
    }
    for (auto* module : modules) {
        snx_recompiled_invalidate(module, addr, size, &InvalidatedRangeThunk, &context);
    }
    MergeInvalidatedRanges(invalidated_ranges);
    // Remove every cached secondary leader in the invalidated function spans. Keep retired native
    // code loaded: another guest thread may still be executing it.
    std::sort(invalidated_ranges.begin(), invalidated_ranges.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& range : invalidated_ranges) {
        if (merged.empty() || merged.back().second < range.first) {
            merged.push_back(range);
        } else {
            merged.back().second = std::max(merged.back().second, range.second);
        }
    }
    if (!merged.empty()) {
        std::unique_lock lock{m_secondary_lock};
        for (auto it = m_secondary.begin(); it != m_secondary.end();) {
            const auto range = std::upper_bound(merged.begin(), merged.end(), it->first,
                                                [](std::uint64_t address, const auto& candidate) {
                                                    return address < candidate.first;
                                                });
            const bool stale = range != merged.begin() && it->first < std::prev(range)->second;
            it = stale ? m_secondary.erase(it) : std::next(it);
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
