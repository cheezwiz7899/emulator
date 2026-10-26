#include "core/arm/scarlet/scarlet_code_manager.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <thread>
using namespace Core::Scarlet;

static uint8_t *mem;
static int fails = 0;
static std::atomic<int> undefs{0};
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      std::printf("FAIL line %d: %s\n", __LINE__, #c);                         \
      fails++;                                                                 \
    }                                                                          \
  } while (0)
struct Src : CodeSource {
  bool ExecRangeAround(uint64_t pc, uint64_t, uint64_t &lo,
                       uint64_t &hi) override {
    if (pc >= 0x2000000 && pc < 0x2001000) {
      lo = 0x2000000;
      hi = 0x2001000;
      return true;
    }
    if (pc >= 0x10000 && pc < 0x11000) {
      lo = 0x10000;
      hi = 0x11000;
      return true;
    }
    return false;
  }
  void Read(uint64_t a, void *d, size_t n) override {
    std::memcpy(d, mem + a, n);
  }
};
static void put(uint64_t at, std::initializer_list<uint32_t> w) {
  uint32_t *p = (uint32_t *)(mem + at);
  for (auto x : w)
    *p++ = x;
}
static void OnSvc(CPU *, uint32_t) {}
static void OnUndef(CPU *c, uint64_t pc, uint32_t) {
  undefs++;
  c->pc = 0;
}
static void OnBrk(CPU *c, uint64_t, uint32_t) { c->pc = 0; }

static uint64_t run(CodeManager &m, RtAPI &, int idx, uint64_t pc) {
  CPU cpu{};
  cpu.mem = mem;
  cpu.x[31] = 0x800000 + idx * 0x10000;
  cpu.x[30] = 0;
  cpu.thread = nullptr;
  if (!m.Bind(&cpu)) {
    std::printf("bind failed\n");
    return ~0ull;
  }
  m.Run(&cpu, pc, 0);
  return cpu.x[0];
}

int main() {
  mem = (uint8_t *)mmap(nullptr, 64u << 20, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  // primary A
  put(0x10000, {0xd2a04009, 0xa9bf7ffe, 0xd63f0120, 0xa8c17ffe, 0x91019000,
                0xd65f03c0});             // call B, +100
  put(0x10100, {0x910FA000, 0xd65f03c0}); // A2: +1000
  put(0x10200, {0xd2a06009, 0xa9bf7ffe, 0xd63f0120, 0xa8c17ffe,
                0xd65f03c0}); // call 0x3000000 (not exec)
  // secondary B
  put(0x2000000, {0x91000400, 0xa9bf7ffe, 0x9400003e, 0x940FC000 | 0,
                  0xa8c17ffe, 0xd65f03c0});
  // patch the bl-to-A2 word properly
  {
    uint32_t w = 0x94000000 | (((0x10100 - 0x200000c) >> 2) & 0x3ffffff);
    put(0x200000c, {w});
  }
  put(0x2000100, {0x91002800, 0xd65f03c0}); // B2: +10

  Src src;
  CodeManager mgr(src);
  std::string err;
  if (!mgr.Initialize({{0x10000, 0x1000}}, nullptr, err)) {
    std::printf("init: %s\n", err.c_str());
    return 1;
  }
  mgr.Rt().svc = OnSvc;
  mgr.Rt().undef = OnUndef;
  mgr.Rt().brk = OnBrk;

  uint64_t r = run(mgr, mgr.Rt(), 0, 0x10000);
  std::printf("call into secondary: x0=%llu modules=%zu fns=%zu\n",
              (unsigned long long)r, mgr.SecondaryModuleCount(),
              mgr.SecondaryFunctionCount());
  CHECK(r == 1111);
  CHECK(mgr.SecondaryModuleCount() == 1);

  r = run(mgr, mgr.Rt(), 0, 0x10000);
  CHECK(r == 1111);
  CHECK(mgr.SecondaryModuleCount() == 1); // cache hit, no recompile

  // Invalidating code whose bytes did not change (a permission flip, a remap, a spurious IC IVAU)
  // must leave the primary module and the cached secondary code in use.
  mgr.Invalidate(0x10000, 0x1000);
  r = run(mgr, mgr.Rt(), 0, 0x10000);
  CHECK(r == 1111);
  CHECK(mgr.SecondaryModuleCount() == 1); // A stayed AOT; no live copy was compiled

  // non-executable target -> undef hook
  run(mgr, mgr.Rt(), 0, 0x10200);
  CHECK(undefs == 1);

  // IC IVAU must retire a primary AOT function and route its next entry to live
  // guest memory.
  r = run(mgr, mgr.Rt(), 0, 0x10100);
  CHECK(r == 1000);
  put(0x10100, {0x91000800, 0xd65f03c0}); // add x0, x0, #2; ret
  {
    CPU cpu{};
    cpu.mem = mem;
    cpu.x[31] = 0x800000;
    cpu.x[30] = 0;
    CHECK(mgr.Bind(&cpu));
    cpu.rt->code_changed(&cpu, 0x10100);
    mgr.Run(&cpu, 0x10100, 0);
    CHECK(cpu.x[0] == 2);
    put(0x10100, {0x910FA000, 0xd65f03c0}); // restore add x0, x0, #1000; ret
    cpu.rt->code_changed(&cpu, 0x10100);
  }

  // invalidate + new code at the same place: recompiled afresh
  mgr.Invalidate(0x2000000, 0x1000);
  put(0x2000000, {0x91000800}); // add x0,x0,#2
  r = run(mgr, mgr.Rt(), 0, 0x10000);
  std::printf("after invalidate: x0=%llu modules=%zu\n", (unsigned long long)r,
              mgr.SecondaryModuleCount());
  CHECK(r == 1112);
  // modules so far: B, the live copy of A2 (changed), B again, the live copy of A2 (restored)
  CHECK(mgr.SecondaryModuleCount() == 4);

  // 4 threads missing the same new range at once: compiled exactly once
  mgr.Invalidate(0x2000000, 0x1000);
  put(0x2000000, {0x91000c00}); // +3
  std::atomic<int> go{0};
  uint64_t res[4]{};
  std::thread t[4];
  for (int i = 0; i < 4; i++)
    t[i] = std::thread([&, i] {
      while (!go) {
      }
      res[i] = run(mgr, mgr.Rt(), i + 1, 0x10000);
    });
  go = 1;
  for (auto &x : t)
    x.join();
  for (int i = 0; i < 4; i++)
    CHECK(res[i] == 1113);
  std::printf("4 threads: modules=%zu\n", mgr.SecondaryModuleCount());
  CHECK(mgr.SecondaryModuleCount() == 5); // exactly one more: the new B
  // Invalidating the whole address space ends (the per-address probe count must not wrap) and
  // leaves everything runnable: code is compiled again on demand.
  mgr.Invalidate(0, ~0ull);
  CHECK(run(mgr, mgr.Rt(), 0, 0x10000) == 1113);

  // Invalidate never waits for a compile, and a compile racing it never publishes stale code:
  // four threads run the secondary code while another invalidates it over and over.
  {
    std::atomic<bool> stop{false};
    std::atomic<int> wrong{0};
    std::thread runners[4];
    for (int i = 0; i < 4; i++)
      runners[i] = std::thread([&, i] {
        while (!stop)
        {
          const auto v = run(mgr, mgr.Rt(), i + 1, 0x10000);
          if (v != 1113)
            wrong++;
        }
      });
    for (int i = 0; i < 20; i++) {
      mgr.Invalidate(0x2000000, 0x1000);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    stop = true;
    for (auto &x : runners)
      x.join();
    std::printf("invalidate race: wrong=%d\n", wrong.load());
    CHECK(wrong == 0);
  }
  std::printf("RESULT %s\n", fails ? "FAIL" : "PASS");
  return fails;
}
