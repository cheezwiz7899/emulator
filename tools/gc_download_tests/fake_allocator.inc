// SPDX-License-Identifier: GPL-2.0-or-later
// Included in the generated pool declaration. Only allocation/deferred-release are fake;
// RequestGCDownload's declaration, cached handle, mutex and body come from production.
enum class Failure { None, DeviceMemory, HostMemory, HostAllocation, Other };
Failure failure{};
std::size_t allocations{}, releases{};
std::vector<std::unique_ptr<std::vector<u8>>> buffers;

StagingBufferRef Request(std::size_t size, MemoryUsage usage, bool deferred) {
    Check(usage == MemoryUsage::Download && deferred, "GC allocation must stay deferred");
    switch (std::exchange(failure, Failure::None)) {
    case Failure::DeviceMemory: throw vk::Exception{VK_ERROR_OUT_OF_DEVICE_MEMORY};
    case Failure::HostMemory: throw vk::Exception{VK_ERROR_OUT_OF_HOST_MEMORY};
    case Failure::HostAllocation: throw std::bad_alloc{};
    case Failure::Other: throw vk::Exception{VK_ERROR_DEVICE_LOST};
    case Failure::None: break;
    }
    auto data = std::make_unique<std::vector<u8>>(size);
    const std::span<u8> mapped{*data};
    buffers.push_back(std::move(data));
    ++allocations;
    return {allocations, 0, mapped, usage, 0, allocations};
}

void FreeDeferred(StagingBufferRef& ref) {
    Check(ref.index != 0, "Only a valid deferred allocation can be released");
    ++releases;
}
