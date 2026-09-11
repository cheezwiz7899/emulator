// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace Shader::Maxwell {

struct PredecodedInstruction {
    u32 location{};
    u64 instruction{};
    Opcode opcode{};

    bool operator==(const PredecodedInstruction&) const noexcept = default;
};

void Translate(Environment& env, IR::Block* block, u32 location_begin, u32 location_end,
               std::span<const PredecodedInstruction> predecoded = {});

} // namespace Shader::Maxwell
