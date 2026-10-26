// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// scarlet_code_manager.h pulls in the recompiler's GNU C header (and drops its macros); it must
// come before the Citron headers.
#include "core/arm/scarlet/scarlet_code_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include <fmt/format.h>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "core/arm/scarlet/arm_scarlet_64.h"
#include "core/arm/scarlet/guest_coroutine.h"
#include "core/arm/scarlet/scarlet_fault_handler.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_memory_block.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/memory.h"

namespace Core {

namespace {

/// Guest code as the process's page table sees it.
class ProcessCodeSource final : public Scarlet::CodeSource {
public:
    explicit ProcessCodeSource(Kernel::KProcess* process) : m_process{process} {}

    bool ExecRangeAround(u64 pc, u64 max_size, u64& lo, u64& hi) override {
        auto& table = m_process->GetPageTable().GetBasePageTable();
        const auto query = [&](u64 address, Kernel::Svc::MemoryInfo& out) {
            Kernel::KMemoryInfo info{};
            Kernel::Svc::PageInfo page{};
            if (!R_SUCCEEDED(table.QueryInfo(std::addressof(info), std::addressof(page),
                                             Kernel::KProcessAddress{address}))) {
                return false;
            }
            out = info.GetSvcMemoryInfo();
            return true;
        };
        const auto executable = [](const Kernel::Svc::MemoryInfo& info) {
            return (info.permission & Kernel::Svc::MemoryPermission::Execute) ==
                   Kernel::Svc::MemoryPermission::Execute;
        };

        Kernel::Svc::MemoryInfo block{};
        if (!query(pc, block) || !executable(block)) {
            return false;
        }
        lo = block.base_address;
        hi = lo + block.size;
        // Blocks with equal permissions are not always merged: walk to the contiguous run, but
        // no further than half the window either way.
        const u64 reach = max_size / 2;
        Kernel::Svc::MemoryInfo next{};
        while (lo > 0 && pc - lo < reach && query(lo - 1, next) && executable(next)) {
            lo = next.base_address;
        }
        while (hi - pc < reach && query(hi, next) && executable(next)) {
            hi = next.base_address + next.size;
        }
        if (pc - lo > reach) {
            lo = pc - reach;
        }
        if (hi - pc > reach) {
            hi = pc + reach;
        }
        lo &= ~u64{3};
        hi &= ~u64{3};
        return pc >= lo && pc < hi;
    }

    void Read(u64 addr, void* dst, size_t size) override {
        m_process->GetMemory().ReadBlock(addr, dst, size);
    }

private:
    Kernel::KProcess* m_process;
};

// Everything recompiled for one process, shared by its four core interfaces.
struct SharedCode {
    explicit SharedCode(Kernel::KProcess* process) : source{process}, manager{source} {
        last_summary_ns.store(HostNowNs(), std::memory_order_relaxed);
    }

    static std::uint64_t HostNowNs() {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now().time_since_epoch())
                                              .count());
    }

    ProcessCodeSource source; // declared first: outlives the manager
    Scarlet::CodeManager manager;
    std::mutex lock;
    bool initialized{};
    bool failed{};
    std::atomic<std::uint64_t> run_calls{};
    std::atomic<std::uint64_t> run_host_ns{};
    std::atomic<std::uint64_t> max_run_ns{};
    std::atomic<std::uint64_t> last_summary_ns{};
    std::atomic<std::uint64_t> reported_run_calls{};
    std::atomic<std::uint64_t> reported_run_host_ns{};
    std::atomic<std::uint64_t> last_guest_pc{};
    std::atomic<std::uint64_t> last_guest_thread{};
    std::atomic<std::uint64_t> last_halt{};
    std::atomic<std::size_t> last_core{};
    std::array<std::atomic<std::uint64_t>, 6> halt_counts{};
};

std::mutex g_registry_lock;
std::map<const Kernel::KProcess*, std::weak_ptr<SharedCode>> g_registry;

std::shared_ptr<SharedCode> SharedCodeFor(Kernel::KProcess* process) {
    std::scoped_lock lk{g_registry_lock};
    // Entries of finished processes would otherwise stay for the life of Citron.
    for (auto it = g_registry.begin(); it != g_registry.end();) {
        it = it->second.expired() ? g_registry.erase(it) : std::next(it);
    }
    auto& slot = g_registry[process];
    if (auto existing = slot.lock()) {
        return existing;
    }
    auto created = std::make_shared<SharedCode>(process);
    slot = created;
    return created;
}

std::atomic<Timing::CoreTiming*> s_core_timing{nullptr};

inline void PauseProcessor() {
#if defined(_MSC_VER)
    YieldProcessor();
#elif defined(__x86_64__) || defined(_M_X64)
    asm volatile("pause");
#elif defined(__aarch64__) || defined(_M_ARM64)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

constexpr u32 PstateFlagsMask = 0xF0000000;

} // namespace

struct ArmScarlet64::Impl {
    Impl(System& system_, Kernel::KProcess* process_, std::size_t core_index_)
        : system{system_}, process{process_}, core_index{core_index_},
          shared{SharedCodeFor(process_)} {}

    ~Impl() {
        if (epoch_registered) {
            shared->manager.RemoveCoroutineEpoch(&co_epoch);
        }
    }

    System& system;
    Kernel::KProcess* process;
    std::size_t core_index;
    std::shared_ptr<SharedCode> shared;

    CPU cpu{};
    bool bound{};
    bool first_dispatch_logged{};
    u32 pstate_rest{}; // PSTATE bits outside NZCV, kept so a context round-trips unchanged
    u32 wfe_count{};
    u32 svc{};
    u64 brk_pc{};
    HaltReason halt{};

    Scarlet::GuestCoroutine co;

    // The code manager's epoch when the current coroutine started, UINT64_MAX while it holds no
    // frames. The manager frees retired native code only when no coroutine older than the
    // retirement can still return into it.
    std::atomic<u64> co_epoch{std::numeric_limits<u64>::max()};
    bool epoch_registered{};

    // What the suspended coroutine is waiting to continue: the same thread, at the same pc.
    Kernel::KThread* resume_thread{};
    u64 resume_pc{};

    static void Body(void* arg) {
        auto* self = static_cast<Impl*>(arg);
        // Runs until the guest returns to address 0 (the stop address). Real threads end with the
        // ExitThread SVC and never get here.
        self->shared->manager.Run(&self->cpu, self->cpu.pc, 0);
        self->halt = HaltReason::PrefetchAbort;
    }

    void Halt(HaltReason reason) {
        halt = reason;
        co.Yield();
    }

    // The runtime hooks. They run inside generated code, on the coroutine's stack.
    static Impl* From(CPU* c) {
        return static_cast<Impl*>(c->thread);
    }
    static void OnSvc(CPU* c, uint32_t imm) {
        auto* self = From(c);
        self->wfe_count = 0;
        self->svc = imm;
        self->Halt(HaltReason::SupervisorCall);
    }
    static void OnPoll(CPU* c) {
        // Set by SignalInterrupt() from another host thread. Generated code has already stored pc
        // and written its registers back (a runtime call invalidates its register cache).
        __atomic_store_n(&c->poll, 0u, __ATOMIC_RELAXED);
        auto* self = From(c);
        self->wfe_count = 0;
        self->Halt(HaltReason::BreakLoop);
    }
    static uint64_t OnTicks() {
        if (auto* timing = s_core_timing.load(std::memory_order_relaxed)) {
            return timing->GetClockTicks();
        }
        return 0;
    }
    static void OnWfe(CPU* c) {
        auto* self = From(c);
        // If an interrupt was signaled by the kernel, yield immediately.
        if (__atomic_load_n(&c->poll, __ATOMIC_RELAXED)) {
            __atomic_store_n(&c->poll, 0u, __ATOMIC_RELAXED);
            self->wfe_count = 0;
            self->Halt(HaltReason::BreakLoop);
            return;
        }
        // Adaptive spin-yielding:
        // Brief hardware pause allows spinlocks held by concurrent threads on other host
        // cores to resolve without incurring an expensive coroutine unwind and scheduler trip.
        // After 32 iterations, we yield back to Citron's kernel scheduler (BreakLoop)
        // so other ready guest threads scheduled on this physical core can run,
        // completely preventing single-core starvation deadlocks.
        if (++self->wfe_count >= 32) {
            self->wfe_count = 0;
            self->Halt(HaltReason::BreakLoop);
        } else {
            PauseProcessor();
        }
    }
    static void OnCodeChanged(CPU* c, uint64_t addr) {
        auto* self = From(c);
        // IC IVAU: guest code modified at runtime (e.g. UltraCam / ExLaunch hooks).
        // Invalidate the 64-byte CTR_EL0 IminLine cache line across primary and secondary JIT.
        self->shared->manager.Invalidate(addr & ~uint64_t{63}, 64);
    }
    static void OnBrk(CPU* c, uint64_t pc, uint32_t imm) {
        auto* self = From(c);
        LOG_CRITICAL(Core_ARM, "BRK #{:#x} at {:#018x}", imm, pc);
        self->brk_pc = pc;
        c->pc = pc;
        self->Halt(HaltReason::InstructionBreakpoint);
    }
    static void OnUndef(CPU* c, uint64_t pc, uint32_t word) {
        auto* self = From(c);
        // ScarletNX's default miss also reports word=0 when it cannot find or compile a
        // function. Read the guest bytes independently so the log distinguishes that case
        // from actually executing an A64 UDF #0 in the module.
        uint32_t guest_word = 0;
        self->process->GetMemory().ReadBlock(pc, &guest_word, sizeof(guest_word));
        if (guest_word != word) {
            const u64 page_base = pc & ~0xFFFull;
            self->shared->manager.Invalidate(page_base, 0x1000);
            if (self->shared->manager.RunLiveFallback(c, pc)) {
                return;
            }
        }
        const auto& ranges = self->process->GetExecutableRanges();
        bool in_executable_range = false;
        for (const auto& [base, size] : ranges) {
            if (pc >= base && pc - base < size) {
                in_executable_range = true;
                break;
            }
        }
        LOG_CRITICAL(
            Core_ARM,
            "Undefined instruction {:#010x} at {:#018x} (guest memory {:#010x}, LR {:#018x}, "
            "SP {:#018x}, {} a loaded executable segment)",
            word, pc, guest_word, c->x[30], c->x[31], in_executable_range ? "inside" : "outside");
        c->pc = pc;
        self->Halt(HaltReason::PrefetchAbort);
    }
    // A real guest memory fault (not a GPU-tracked page): see scarlet_fault_handler.h. Runs in
    // place of the faulting instruction; the thread is parked and never continues.
    static void OnGuestFault(void* context, u64 host_address, u64 is_write) {
        auto* self = static_cast<Impl*>(context);
        const u64 base = reinterpret_cast<u64>(self->cpu.mem);
        u32 instruction = 0;
        self->process->GetMemory().ReadBlock(self->cpu.pc, &instruction, sizeof(instruction));
        LOG_CRITICAL(Core_ARM,
                     "Guest memory fault ({}) at guest address {:#x} (host {:#x}); last known pc "
                     "{:#018x}; instruction {:#010x}, LR {:#018x}, SP {:#018x}, code range "
                     "{:#018x}-{:#018x}",
                     is_write != 0 ? "write" : "read", host_address - base, host_address,
                     self->cpu.pc, instruction, self->cpu.x[30], self->cpu.x[31], self->cpu.code_lo,
                     self->cpu.code_hi);
        for (u32 first = 0; first < 28; first += 4) {
            LOG_CRITICAL(Core_ARM,
                         "Guest fault registers x{}-x{}: {:#018x} {:#018x} {:#018x} {:#018x}",
                         first, first + 3, self->cpu.x[first], self->cpu.x[first + 1],
                         self->cpu.x[first + 2], self->cpu.x[first + 3]);
        }
        LOG_CRITICAL(Core_ARM, "Guest fault registers x28-x30: {:#018x} {:#018x} {:#018x}",
                     self->cpu.x[28], self->cpu.x[29], self->cpu.x[30]);
        for (;;) {
            self->Halt(HaltReason::PrefetchAbort);
        }
    }

    // Recompiles the process's executable segments once (shared by the four cores). Blocks the
    // calling emulation thread: a first run of a title takes as long as ScarletNX's own
    // "Recompiling" step. Results are cached on disk by content hash.
    bool EnsureModule() {
        auto& sh = *shared;
        std::scoped_lock lk{sh.lock};
        if (sh.initialized) {
            return true;
        }
        if (sh.failed) {
            return false;
        }
        sh.failed = true;

        std::vector<Scarlet::CodeManager::Segment> segments;
        std::vector<Scarlet::CodeManager::ModuleImage> modules;
        for (const auto& [addr, size] : process->GetExecutableRanges()) {
            segments.push_back({addr, size});
            LOG_INFO(Core_ARM, "ScarletNX: executable range {:#018x}-{:#018x} ({} bytes)", addr,
                     addr + size, size);
        }
        if (segments.empty()) {
            LOG_CRITICAL(Core_ARM, "No executable segments recorded for the process");
            return false;
        }
        for (const auto& module : process->GetExecutableModules()) {
            modules.push_back(
                {module.image_address, module.image_size, module.text_offset, module.text_size});
        }
        const auto& build_id_raw = system.GetApplicationProcessBuildID();
        const bool has_build_id = std::any_of(build_id_raw.begin(), build_id_raw.end(),
                                              [](u8 b) { return b != 0; });
        const auto build_id_str =
            has_build_id ? Common::HexToString(build_id_raw, false) : std::string{};

        auto cache = Common::FS::GetCitronPath(Common::FS::CitronPath::CitronDir) /
                     "scarlet" / "modules" / fmt::format("{:016X}", process->GetProgramId());
        if (has_build_id) {
            cache /= build_id_str;
        }
        const auto cache_utf8 = Common::FS::PathToUTF8String(cache);
        LOG_INFO(Core_ARM, "ScarletNX: module cache path: {}", cache_utf8);
        LOG_INFO(Core_ARM, "ScarletNX: recompiling {} executable segment(s)...", segments.size());

        std::string error;
        system.ReportScarletRecompileProgress(false, 0, 0);
        const bool initialized = sh.manager.Initialize(
            segments, cache_utf8.c_str(), error,
            [this](std::size_t done, std::size_t total) {
                system.ReportScarletRecompileProgress(false, done, total);
            },
            modules);
        system.ReportScarletRecompileProgress(true, 0, 0);
        if (!initialized) {
            LOG_CRITICAL(Core_ARM, "ScarletNX recompile failed: {}", error);
            return false;
        }
        s_core_timing.store(&system.CoreTiming(), std::memory_order_relaxed);
        auto& rt = sh.manager.Rt();
        rt.svc = &Impl::OnSvc;
        rt.undef = &Impl::OnUndef;
        rt.brk = &Impl::OnBrk;
        rt.poll = &Impl::OnPoll;
        rt.wfe = &Impl::OnWfe;
        rt.ticks = &Impl::OnTicks;
        rt.code_changed = &Impl::OnCodeChanged;
        sh.initialized = true;
        sh.failed = false;
        return true;
    }

    bool EnsureBound() {
        if (bound) {
            return true;
        }
        if (!EnsureModule()) {
            return false;
        }
        // Guest address a is host byte arena[a]: the same flat mapping dynarmic's fastmem uses.
        cpu.mem = process->GetPageTable().GetBasePageTable().GetImpl().fastmem_arena;
        if (!cpu.mem) {
            LOG_CRITICAL(Core_ARM, "ScarletNX backend needs fastmem (a flat guest mapping)");
            return false;
        }
        cpu.thread = this;
        if (!shared->manager.Bind(&cpu)) {
            LOG_CRITICAL(Core_ARM, "ScarletNX: binding the CPU to the module failed");
            return false;
        }
        shared->manager.AddCoroutineEpoch(&co_epoch);
        epoch_registered = true;
        bound = true;
        return true;
    }
};

ArmScarlet64::ArmScarlet64(System& system, bool uses_wall_clock, Kernel::KProcess* process,
                           std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_impl{std::make_unique<Impl>(system, process, core_index)} {}

ArmScarlet64::~ArmScarlet64() = default;

void ArmScarlet64::Initialize() {
    // Idempotent. Guest writes to pages the GPU caches fault in the fastmem arena; this turns the
    // fault into the same invalidation the NCE backend does.
    InstallScarletFaultHandler(m_impl->system);
}

HaltReason ArmScarlet64::RunThread(Kernel::KThread* thread) {
    auto& s = *m_impl;
    if (!s.EnsureBound()) {
        return HaltReason::PrefetchAbort;
    }
    s.cpu.excl_on = 0;
    s.wfe_count = 0;

    // Fast path: the same thread continues where its last event stopped (the common SVC case).
    // Anything else (another thread, another core, a rewritten pc) starts a fresh run at cpu.pc,
    // which the recompiler's lookup JIT-compiles if pc is not a function entry.
    const bool resumable =
        s.co.IsSuspended() && s.resume_thread == thread && s.resume_pc == s.cpu.pc;
    if (!resumable) {
        s.co.Reset();
        // Read before the first frame exists: a module retired after this cannot have been
        // entered through a route this coroutine took.
        s.co_epoch.store(s.shared->manager.Epoch(), std::memory_order_seq_cst);
        s.co.Start(&Impl::Body, &s);
    }

    s.halt = {};
    SetGuestFaultSink(&s, &Impl::OnGuestFault);
    const auto run_started = std::chrono::steady_clock::now();
    s.co.Resume();
    const auto run_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - run_started)
                                       .count());
    SetGuestFaultSink(nullptr, nullptr);

    if (!s.first_dispatch_logged) {
        s.first_dispatch_logged = true;
        LOG_DEBUG(
            Core_ARM,
            "ScarletNX first dispatch: core={} guest_thread={:#x} pc={:#018x} halt={:#x} run={} us",
            s.core_index, thread->GetThreadId(), s.cpu.pc, static_cast<u64>(s.halt), run_ns / 1000,
            s.co.IsFinished() ? "finished" : "yielded");
    }

    auto& stats = *s.shared;
    stats.run_calls.fetch_add(1, std::memory_order_relaxed);
    stats.run_host_ns.fetch_add(run_ns, std::memory_order_relaxed);
    auto max_ns = stats.max_run_ns.load(std::memory_order_relaxed);
    while (max_ns < run_ns &&
           !stats.max_run_ns.compare_exchange_weak(max_ns, run_ns, std::memory_order_relaxed)) {
    }
    stats.last_guest_pc.store(s.cpu.pc, std::memory_order_relaxed);
    stats.last_guest_thread.store(thread->GetThreadId(), std::memory_order_relaxed);
    stats.last_core.store(s.core_index, std::memory_order_relaxed);
    stats.last_halt.store(static_cast<u64>(s.halt), std::memory_order_relaxed);
    switch (s.halt) {
    case HaltReason::SupervisorCall:
        stats.halt_counts[0].fetch_add(1, std::memory_order_relaxed);
        break;
    case HaltReason::BreakLoop:
        stats.halt_counts[1].fetch_add(1, std::memory_order_relaxed);
        break;
    case HaltReason::PrefetchAbort:
        stats.halt_counts[2].fetch_add(1, std::memory_order_relaxed);
        break;
    case HaltReason::DataAbort:
        stats.halt_counts[3].fetch_add(1, std::memory_order_relaxed);
        break;
    case HaltReason::InstructionBreakpoint:
        stats.halt_counts[4].fetch_add(1, std::memory_order_relaxed);
        break;
    case HaltReason::StepThread:
        stats.halt_counts[5].fetch_add(1, std::memory_order_relaxed);
        break;
    default:
        break;
    }

    const auto now_ns = SharedCode::HostNowNs();
    auto last_summary = stats.last_summary_ns.load(std::memory_order_relaxed);
    if (now_ns - last_summary >= 10'000'000'000ull &&
        stats.last_summary_ns.compare_exchange_strong(last_summary, now_ns,
                                                      std::memory_order_relaxed)) {
        s.shared->manager.PeriodicSaveHints();
        const auto calls = stats.run_calls.load(std::memory_order_relaxed);
        const auto host_ns = stats.run_host_ns.load(std::memory_order_relaxed);
        const auto previous_calls =
            stats.reported_run_calls.exchange(calls, std::memory_order_relaxed);
        const auto previous_host_ns =
            stats.reported_run_host_ns.exchange(host_ns, std::memory_order_relaxed);
        LOG_DEBUG(
            Core_ARM,
            "ScarletNX runtime 10s diagnostic: interval calls={} active_host={} ms; "
            "lifetime calls={} active_host={} ms avg={} us max={} ms; "
            "halts svc={} poll={} prefetch={} data={} brk={} step={}; last core={} thread={:#x} "
            "pc={:#018x} halt={:#x}",
            calls - previous_calls, (host_ns - previous_host_ns) / 1000000, calls,
            host_ns / 1000000, calls ? host_ns / calls / 1000 : 0,
            stats.max_run_ns.load(std::memory_order_relaxed) / 1000000,
            stats.halt_counts[0].load(std::memory_order_relaxed),
            stats.halt_counts[1].load(std::memory_order_relaxed),
            stats.halt_counts[2].load(std::memory_order_relaxed),
            stats.halt_counts[3].load(std::memory_order_relaxed),
            stats.halt_counts[4].load(std::memory_order_relaxed),
            stats.halt_counts[5].load(std::memory_order_relaxed),
            stats.last_core.load(std::memory_order_relaxed),
            stats.last_guest_thread.load(std::memory_order_relaxed),
            stats.last_guest_pc.load(std::memory_order_relaxed),
            stats.last_halt.load(std::memory_order_relaxed));
    }

    if (s.co.IsFinished()) {
        s.co_epoch.store(std::numeric_limits<u64>::max(), std::memory_order_seq_cst);
        s.resume_thread = nullptr; // Body() set the halt reason
    } else {
        s.resume_thread = thread;
        s.resume_pc = s.cpu.pc;
    }
    return s.halt;
}

HaltReason ArmScarlet64::StepThread(Kernel::KThread*) {
    // Recompiled code has no single-step. A debugger session needs the dynarmic backend.
    LOG_WARNING(Core_ARM, "ScarletNX backend: single-step is not supported");
    return HaltReason::StepThread;
}

void ArmScarlet64::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    const CPU& c = m_impl->cpu;
    for (std::size_t i = 0; i < 29; i++) {
        ctx.r[i] = c.x[i];
    }
    ctx.fp = c.x[29];
    ctx.lr = c.x[30];
    ctx.sp = c.x[31];
    ctx.pc = c.pc;
    ctx.pstate = m_impl->pstate_rest | (c.n ? 1u << 31 : 0) | (c.z ? 1u << 30 : 0) |
                 (c.cf ? 1u << 29 : 0) | (c.vf ? 1u << 28 : 0);
    static_assert(sizeof(ctx.v) == sizeof(c.v));
    std::memcpy(ctx.v.data(), c.v, sizeof(c.v));
    ctx.fpcr = c.fpcr;
    ctx.fpsr = c.fpsr;
    ctx.tpidr = c.tpidr;
}

void ArmScarlet64::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    m_impl->resume_thread = nullptr;
    // The suspended frames, if any, are never resumed now: the next RunThread starts afresh.
    m_impl->co_epoch.store(std::numeric_limits<u64>::max(), std::memory_order_seq_cst);
    CPU& c = m_impl->cpu;
    for (std::size_t i = 0; i < 29; i++) {
        c.x[i] = ctx.r[i];
    }
    c.x[29] = ctx.fp;
    c.x[30] = ctx.lr;
    c.x[31] = ctx.sp;
    c.pc = ctx.pc;
    m_impl->pstate_rest = ctx.pstate & ~PstateFlagsMask;
    c.n = (ctx.pstate >> 31) & 1;
    c.z = (ctx.pstate >> 30) & 1;
    c.cf = (ctx.pstate >> 29) & 1;
    c.vf = (ctx.pstate >> 28) & 1;
    std::memcpy(c.v, ctx.v.data(), sizeof(c.v));
    c.fpcr = ctx.fpcr;
    c.fpsr = ctx.fpsr;
    c.tpidr = ctx.tpidr;
}

void ArmScarlet64::SetTpidrroEl0(u64 value) {
    m_impl->cpu.tpidrro = value;
}

void ArmScarlet64::GetSvcArguments(std::span<uint64_t, 8> args) const {
    for (std::size_t i = 0; i < 8; i++) {
        args[i] = m_impl->cpu.x[i];
    }
}

void ArmScarlet64::SetSvcArguments(std::span<const uint64_t, 8> args) {
    for (std::size_t i = 0; i < 8; i++) {
        m_impl->cpu.x[i] = args[i];
    }
}

u32 ArmScarlet64::GetSvcNumber() const {
    return m_impl->svc;
}

void ArmScarlet64::SignalInterrupt(Kernel::KThread*) {
    // Read on every backward branch of generated code; may be called from any host thread.
    __atomic_store_n(&m_impl->cpu.poll, 1u, __ATOMIC_RELAXED);
}

/// The page table and the debugger notify every core of a process, but the compiled code is shared
/// by all of them: only the first core's interface acts, or each event would be handled four times.
bool ArmScarlet64::IsInvalidationLeader() const {
    for (std::size_t i = 0; i < Core::Hardware::NUM_CPU_CORES; ++i) {
        if (const auto* interface = m_impl->process->GetArmInterface(i)) {
            return interface == static_cast<const ArmInterface*>(this);
        }
    }
    return true;
}

void ArmScarlet64::ClearInstructionCache() {
    // A guest `ic iallu` / debugger edit. Primary code is compared with what it was compiled from
    // (a line at a time), so this retires only code whose bytes differ; secondary entries go.
    if (IsInvalidationLeader()) {
        m_impl->shared->manager.Invalidate(0, ~u64{0});
    }
}

void ArmScarlet64::InvalidateCacheRange(u64 addr, std::size_t size) {
    // Called by the page table for every core when code memory is mapped, unmapped or
    // reprotected (an NRO unloaded and another loaded at the same place), and by the debugger and
    // cheats. Cheap when the bytes did not change.
    if (IsInvalidationLeader()) {
        m_impl->shared->manager.Invalidate(addr, size);
    }
}

const Kernel::DebugWatchpoint* ArmScarlet64::HaltedWatchpoint() const {
    return nullptr;
}

void ArmScarlet64::RewindBreakpointInstruction() {
    m_impl->cpu.pc = m_impl->brk_pc;
}

} // namespace Core
