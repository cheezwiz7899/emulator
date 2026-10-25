#include "core/arm/scarlet/scarlet_code_manager.h"
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include "core/arm/scarlet/guest_coroutine.h"
#include "core/arm/scarlet/scarlet_fault_handler.h"
#include "core/core.h"
using namespace Core::Scarlet;

static uint8_t* mem; static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

// stub of the Citron memory API: page 0x400000 is "GPU tracked": unprotect it and retry
int g_invalidations = 0;
bool Core::Memory::Memory::InvalidateNCE(u64 vaddr, std::size_t) {
    if (vaddr == 0x400000) { mprotect(mem + vaddr, 4096, PROT_READ | PROT_WRITE); g_invalidations++; return true; }
    return false;
}
struct Src : CodeSource {
    bool ExecRangeAround(uint64_t, uint64_t, uint64_t&, uint64_t&) override { return false; }
    void Read(uint64_t a, void* d, size_t n) override { std::memcpy(d, mem + a, n); }
};
static void put(uint64_t at, std::initializer_list<uint32_t> w) { uint32_t* p = (uint32_t*)(mem + at); for (auto x : w) *p++ = x; }
struct Ctx { CPU cpu{}; GuestCoroutine co; CodeManager* mgr; int faults = 0; uint64_t addr = 0, write = 9; };
static void Body(void* a) { auto* s = (Ctx*)a; s->mgr->Run(&s->cpu, s->cpu.pc, 0); }
static void FaultFn(void* c, u64 host, u64 is_write) {
    auto* s = (Ctx*)c; s->faults++; s->addr = host - (u64)mem; s->write = is_write;
    for (;;) s->co.Yield();   // park, like ArmScarlet64::OnGuestFault
}
static void OnSvc(CPU*, uint32_t) {}
static void OnUndef(CPU* c, uint64_t, uint32_t) { c->pc = 0; }
static void OnBrk(CPU* c, uint64_t, uint32_t) { c->pc = 0; }

static void start(Ctx& s, uint64_t pc) {
    s.cpu.pc = pc; s.cpu.x[30] = 0; s.cpu.x[31] = 0x800000; s.faults = 0;
    s.co.Reset(); s.co.Start(Body, &s);
    Core::SetGuestFaultSink(&s, &FaultFn);
    s.co.Resume();
    Core::SetGuestFaultSink(nullptr, nullptr);
}

int main() {
    const size_t sz = 64u << 20;
    mem = (uint8_t*)mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    put(0x10000, {0xd2a00801, 0xd2824682, 0xf9000022, 0xf9400020, 0xd65f03c0});
    put(0x10100, {0xd2a00a01, 0xd28000e2, 0xf9000022, 0xd65f03c0});
    put(0x10200, {0xd2a00a01, 0xf9400020, 0xd65f03c0});
    Src src; CodeManager mgr(src); std::string err;
    if (!mgr.Initialize({{0x10000, 0x1000}}, nullptr, err)) { std::printf("init: %s\n", err.c_str()); return 1; }
    mgr.Rt().svc = OnSvc; mgr.Rt().undef = OnUndef; mgr.Rt().brk = OnBrk;

    Core::System system; system.dm.buffer = {mem, sz};
    Core::InstallScarletFaultHandler(system);
    Ctx s; s.mgr = &mgr; s.cpu.mem = mem; CHECK(mgr.Bind(&s.cpu));
    s.cpu.thread = &s;

    // 1. write to a GPU-tracked (read-only) page: handler invalidates, instruction retries
    mprotect(mem + 0x400000, 4096, PROT_READ);
    start(s, 0x10000);
    std::printf("tracked write: finished=%d x0=%llx invalidations=%d faults=%d\n", s.co.IsFinished(), (unsigned long long)s.cpu.x[0], g_invalidations, s.faults);
    CHECK(s.co.IsFinished()); CHECK(s.cpu.x[0] == 0x1234); CHECK(g_invalidations == 1); CHECK(s.faults == 0);

    // 2. write to an unmapped (PROT_NONE) page: redirected to the sink on the coroutine stack, parked
    mprotect(mem + 0x500000, 4096, PROT_NONE);
    start(s, 0x10100);
    std::printf("untracked write: suspended=%d faults=%d addr=%llx write=%llu\n", s.co.IsSuspended(), s.faults, (unsigned long long)s.addr, (unsigned long long)s.write);
    CHECK(s.co.IsSuspended()); CHECK(s.faults == 1); CHECK(s.addr == 0x500000); CHECK(s.write == 1);

    // 3. read fault likewise
    s.cpu.x[1] = 0;
    start(s, 0x10200);
    std::printf("untracked read: suspended=%d faults=%d addr=%llx write=%llu\n", s.co.IsSuspended(), s.faults, (unsigned long long)s.addr, (unsigned long long)s.write);
    CHECK(s.co.IsSuspended()); CHECK(s.faults == 1); CHECK(s.addr == 0x500000); CHECK(s.write == 0);

    // 4. the process is still alive and the handler still works: tracked page again
    mprotect(mem + 0x400000, 4096, PROT_READ); g_invalidations = 0;
    start(s, 0x10000);
    CHECK(s.co.IsFinished()); CHECK(g_invalidations == 1);
    std::printf("RESULT %s\n", fails ? "FAIL" : "PASS");
    return fails;
}
