// SPDX-License-Identifier: GPL-2.0-or-later
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <latch>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// Inject control-block OOM before the allocator is allowed to reserve a buffer.
thread_local bool fail_next_new{};
void* operator new(std::size_t size) {
    if (std::exchange(fail_next_new, false)) throw std::bad_alloc{};
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using VkBuffer = u64;
using VkDeviceSize = u64;
enum class MemoryUsage { Download };
constexpr std::size_t operator""_MiB(unsigned long long value) { return value * 1024 * 1024; }
constexpr int VK_ERROR_OUT_OF_DEVICE_MEMORY = 1;
constexpr int VK_ERROR_OUT_OF_HOST_MEMORY = 2;
constexpr int VK_ERROR_DEVICE_LOST = 3;
namespace vk {
class Exception : public std::exception {
public:
    explicit Exception(int result_) : result{result_} {}
    int GetResult() const { return result; }
private:
    int result;
};
}
void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error{message};
}
#include "gc_request.inc"

void LifetimeAndReuse() {
    StagingBufferPool pool;
    auto texture = pool.RequestGCDownload(64);
    Check(bool(texture), "Initial request failed");
    const auto index = texture->index;
    Check(!pool.RequestGCDownload(32), "Busy allocation was reused at offset zero");
    Check(!pool.RequestGCDownload(128), "Busy allocation was replaced during GPU work");
    Check(pool.allocations == 1 && pool.releases == 0, "Busy requests changed allocation state");
    auto retained = texture;
    texture.reset();
    Check(!pool.RequestGCDownload(32), "A copied batch handle lost its reservation");
    retained.reset();
    auto buffer = pool.RequestGCDownload(32);
    Check(buffer && buffer->index == index && buffer->offset == 0,
          "Completed batch did not reuse the allocation");
    Check(pool.allocations == 1 && pool.releases == 0, "Reuse allocated or released storage");
    buffer.reset();
    auto grown = pool.RequestGCDownload(128);
    Check(grown && grown->mapped_span.size() >= 128 && grown->index != index,
          "Idle storage did not grow");
    Check(pool.allocations == 2 && pool.releases == 1, "Growth did not retire old storage once");
}

void ConcurrentFinishAndConsumption() {
    StagingBufferPool pool;
    std::latch recorded{1}, finish_allowed{1}, finished{1}, consume_allowed{1};
    std::exception_ptr error;
    u64 original_index{};
    u8 consumed{};
    std::thread texture([&] {
        try {
            const auto staging = pool.RequestGCDownload(64);
            Check(bool(staging), "Texture batch request failed");
            original_index = staging->index;
            auto map = *staging;
            recorded.count_down();
            finish_allowed.wait(); // Copies remain in flight until the mock Finish completes.
            map.mapped_span[0] = 0x7b;
            finished.count_down();
            consume_allowed.wait();
            consumed = map.mapped_span[0]; // Reservation must extend beyond Finish to this read.
        } catch (...) {
            error = std::current_exception();
            recorded.count_down();
            finished.count_down();
        }
    });
    recorded.wait();
    const bool blocked_during_copies = !pool.RequestGCDownload(64);
    const bool blocked_growth = !pool.RequestGCDownload(128);
    finish_allowed.count_down();
    finished.wait();
    const bool blocked_after_finish = !pool.RequestGCDownload(64);
    consume_allowed.count_down();
    texture.join();
    if (error) std::rethrow_exception(error);
    Check(blocked_during_copies && blocked_growth && blocked_after_finish,
          "Concurrent buffer GC reused texture staging before mapped consumption");
    Check(consumed == 0x7b, "Texture mapped result was corrupted");
    auto buffer = pool.RequestGCDownload(64);
    Check(buffer && buffer->index == original_index && pool.allocations == 1,
          "Concurrent batch completion did not preserve allocation reuse");
}

void FailurePreservesAllocation() {
    StagingBufferPool pool;
    auto initial = pool.RequestGCDownload(64);
    const auto index = initial->index;
    initial.reset();
    for (auto failure : {StagingBufferPool::Failure::DeviceMemory,
                         StagingBufferPool::Failure::HostMemory,
                         StagingBufferPool::Failure::HostAllocation}) {
        pool.failure = failure;
        Check(!pool.RequestGCDownload(128), "OOM must defer dirty readback");
        auto reused = pool.RequestGCDownload(64);
        Check(reused && reused->index == index, "OOM lost the cached allocation");
        Check(pool.allocations == 1 && pool.releases == 0, "OOM retired live storage");
    }
    pool.failure = StagingBufferPool::Failure::Other;
    bool propagated = false;
    try { (void)pool.RequestGCDownload(128); }
    catch (const vk::Exception& e) { propagated = e.GetResult() == VK_ERROR_DEVICE_LOST; }
    Check(propagated, "Non-OOM Vulkan errors must propagate");
    Check(bool(pool.RequestGCDownload(64)), "Exception left the request mutex locked");
}

void ControlBlockFailure() {
    StagingBufferPool pool;
    fail_next_new = true;
    Check(!pool.RequestGCDownload(64), "Control-block OOM must fail the request");
    Check(pool.allocations == 0 && pool.releases == 0,
          "Control-block OOM stranded a deferred allocation");
    Check(bool(pool.RequestGCDownload(64)), "Recovery after control-block OOM failed");
}

void Limits() {
    StagingBufferPool pool;
    Check(!pool.RequestGCDownload(0) && !pool.RequestGCDownload(32_MiB + 1),
          "Invalid sizes were allocated");
    auto max = pool.RequestGCDownload(32_MiB);
    Check(max && max->mapped_span.size() == 32_MiB, "32 MiB limit was rejected");
    Check(pool.allocations == 1, "Invalid requests allocated storage");
}

int main() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"lifetime, aliases, reuse and growth", LifetimeAndReuse},
        {"concurrent copies, Finish and mapped consumption", ConcurrentFinishAndConsumption},
        {"allocation failure and exception recovery", FailurePreservesAllocation},
        {"control-block failure before reservation", ControlBlockFailure},
        {"bounded allocation limits", Limits},
    };
    unsigned failed = 0;
    for (auto [name, test] : tests) {
        try { test(); std::printf("PASS: %s\n", name); }
        catch (const std::exception& e) {
            ++failed;
            std::fprintf(stderr, "FAIL: %s: %s\n", name, e.what());
        }
    }
    return failed ? 1 : 0;
}
