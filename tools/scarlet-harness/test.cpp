#include <scarletnx/snx_recompiler.h>
#undef LIKELY
#undef UNLIKELY
#undef YIELD
#undef MP
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <sys/mman.h>
#include "core/arm/scarlet/guest_coroutine.h"

using Core::Scarlet::GuestCoroutine;
struct Ctx { CPU cpu{}; GuestCoroutine co; snx_recompiled* m{}; int svc{}; int polls{}; int reason{}; std::atomic<int> started{0}; };
static void Body(void* a) { auto* s = (Ctx*)a; snx_recompiled_run(s->m, &s->cpu, s->cpu.pc, 0); s->reason = 99; }
static void OnSvc(CPU* c, uint32_t imm) { auto* s = (Ctx*)c->thread; s->svc = imm; s->reason = 1; s->co.Yield(); }
static void OnPoll(CPU* c) { auto* s = (Ctx*)c->thread; __atomic_store_n(&c->poll, 0u, __ATOMIC_RELAXED); s->polls++; s->reason = 2; s->co.Yield(); }
static void OnBrk(CPU* c, uint64_t pc, uint32_t) { c->pc = 0; }
static void OnUndef(CPU* c, uint64_t pc, uint32_t w) { std::printf("UNDEF %llx %08x\n", (unsigned long long)pc, w); c->pc = 0; }
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    const size_t memsz = 64u << 20;
    auto* mem = (uint8_t*)mmap(nullptr, memsz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint32_t prog[] = {
        0xd2800000, 0x91000400, 0xd4000841, 0xf100141f, 0x54ffffab, 0xd65f03c0,   // 0x10000: count with svc #0x42
    };
    uint32_t straight[] = {0x91000421, 0x91000421, 0x91000421, 0x91000421, 0xd65f03c0}; // 0x10100
    uint32_t spin[] = {0x14000000};                                                  // 0x10200: b .
    std::memcpy(mem + 0x10000, prog, sizeof prog);
    std::memcpy(mem + 0x10100, straight, sizeof straight);
    std::memcpy(mem + 0x10200, spin, sizeof spin);
    std::vector<uint8_t> blob(0x1000);
    std::memcpy(blob.data(), mem + 0x10000, 0x300);

    auto* r = snx_recompiler_new(SNX_RECOMPILER_FLAG_LLVM);
    CHECK(r);
    snx_code_range range{0x10000, blob.data(), 0x1000};
    uint64_t hints[] = {0x10000, 0x10100, 0x10200};
    auto* m = snx_recompile_ranges(r, &range, 1, hints, 3);
    if (!m) { std::printf("recompile: %s\n", snx_recompiler_error(r)); return 1; }
    RtAPI rt; snx_runtime_defaults(&rt);
    rt.svc = OnSvc; rt.poll = OnPoll; rt.brk = OnBrk; rt.undef = OnUndef;

    auto* s = new Ctx; s->m = m;
    s->cpu.mem = mem; s->cpu.rt = &rt; s->cpu.thread = s;
    CHECK(snx_recompiled_bind(m, &s->cpu) == 0);

    // 1. SVC yields and resumes mid-function (fast path), 5 times
    s->cpu.pc = 0x10000; s->cpu.x[30] = 0; s->cpu.x[31] = 0x800000;
    s->co.Start(Body, s);
    int n = 0;
    for (;;) {
        s->co.Resume();
        if (s->co.IsFinished()) break;
        n++;
        CHECK(s->svc == 0x42);
        CHECK(s->cpu.x[0] == (uint64_t)n);
        CHECK(s->cpu.pc == 0x1000c);
        if (n == 3) { // resume from a different host thread
            std::thread t([&] { s->co.Resume(); });
            t.join();
            if (s->co.IsFinished()) { std::printf("finished early on thread resume\n"); break; }
            n++; CHECK(s->cpu.x[0] == (uint64_t)n);
        }
        if (n > 20) break;
    }
    std::printf("test1: svc yields=%d reason(finish)=%d x0=%llu\n", n, s->reason, (unsigned long long)s->cpu.x[0]);
    CHECK(n == 5); CHECK(s->reason == 99); CHECK(s->cpu.x[0] == 5);

    // 2. fresh entry at a non-leader pc (JIT lookup): expect two adds
    s->cpu.x[1] = 0; s->cpu.pc = 0x10108; s->cpu.x[30] = 0;
    s->co.Reset(); s->co.Start(Body, s); s->reason = 0;
    s->co.Resume();
    std::printf("test2: x1=%llu finished=%d reason=%d\n", (unsigned long long)s->cpu.x[1], s->co.IsFinished(), s->reason);
    CHECK(s->co.IsFinished()); CHECK(s->cpu.x[1] == 2);

    // 2b. fresh entry at the pc after an svc (what a cross-core resume does)
    s->cpu.x[0] = 5; s->cpu.pc = 0x1000c; s->cpu.x[30] = 0; s->co.Reset(); s->co.Start(Body, s); s->reason = 0;
    s->co.Resume();
    std::printf("test2b: finished=%d reason=%d x0=%llu\n", s->co.IsFinished(), s->reason, (unsigned long long)s->cpu.x[0]);
    CHECK(s->co.IsFinished());

    // 3. poll preempts a tight loop; interrupt set from another thread
    s->cpu.pc = 0x10200; s->cpu.x[30] = 0; s->co.Reset(); s->co.Start(Body, s); s->reason = 0; s->polls = 0;
    std::thread killer([&] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); __atomic_store_n(&s->cpu.poll, 1u, __ATOMIC_RELAXED); });
    s->co.Resume();
    killer.join();
    std::printf("test3: polls=%d reason=%d suspended=%d pc=%llx\n", s->polls, s->reason, s->co.IsSuspended(), (unsigned long long)s->cpu.pc);
    CHECK(s->polls == 1); CHECK(s->reason == 2); CHECK(s->co.IsSuspended());
    // resume after poll: still spinning; interrupt again
    std::thread killer2([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); __atomic_store_n(&s->cpu.poll, 1u, __ATOMIC_RELAXED); });
    s->co.Resume(); killer2.join();
    CHECK(s->polls == 2);
    std::printf("RESULT %s\n", fails ? "FAIL" : "PASS");
    return fails;
}
