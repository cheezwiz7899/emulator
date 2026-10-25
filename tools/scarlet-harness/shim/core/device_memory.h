#pragma once
#include "common/common_types.h"
namespace Common { struct HostMemory { u8* base; std::size_t size; u8* VirtualBasePointer() { return base; } std::size_t VirtualSize() { return size; } }; }
namespace Core { struct DeviceMemory { Common::HostMemory buffer; }; }
