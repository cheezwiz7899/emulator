#pragma once
#include <cstddef>
#include <sys/mman.h>
namespace Common {
template <typename T> class VirtualBuffer {
public:
    VirtualBuffer() = default;
    explicit VirtualBuffer(std::size_t count) : n{count} {
        p = static_cast<T*>(mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    }
    ~VirtualBuffer() { if (p) munmap(p, n * sizeof(T)); }
    T* data() { return p; }
    std::size_t size() const { return n; }
private: T* p{}; std::size_t n{};
};
}
