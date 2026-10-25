#pragma once
#include "common/common_types.h"
namespace Core::Memory {
constexpr u64 CITRON_PAGESIZE = 0x1000, CITRON_PAGEMASK = 0xfff;
class Memory { public: bool InvalidateNCE(u64 vaddr, std::size_t size); };
}
