#pragma once
#include "core/memory.h"
#include "core/device_memory.h"
namespace Core { class System { public: Core::DeviceMemory dm; Memory::Memory mem; Core::DeviceMemory& DeviceMemory() { return dm; } Memory::Memory& ApplicationMemory() { return mem; } }; }
