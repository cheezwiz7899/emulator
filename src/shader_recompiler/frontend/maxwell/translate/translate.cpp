// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/location.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/translate.h"

namespace Shader::Maxwell {

void Translate(Environment& env, IR::Block* block, u32 location_begin, u32 location_end,
               std::span<const PredecodedInstruction> predecoded) {
    if (location_begin != location_end) {
        TranslatorVisitor visitor{env, *block};
        auto decoded = std::lower_bound(predecoded.begin(), predecoded.end(), location_begin,
                                        [](const PredecodedInstruction& entry, u32 location) {
                                            return entry.location < location;
                                        });
        for (Location pc = location_begin; pc != location_end; ++pc) {
            u64 insn{};
            Opcode opcode{};
            if (predecoded.empty()) {
                insn = env.ReadInstruction(pc.Offset());
                opcode = Decode(insn);
            } else {
                if (decoded == predecoded.end() || decoded->location != pc.Offset()) {
                    throw LogicError("Missing predecoded instruction at {:x}", pc.Offset());
                }
                insn = decoded->instruction;
                opcode = decoded->opcode;
                ++decoded;
            }
            switch (opcode) {
#define INST(name, cute, mask) case Opcode::name: visitor.name(insn); break;
#include "shader_recompiler/frontend/maxwell/maxwell.inc"
#undef OPCODE
            }
        }
    }
}

} // namespace Shader::Maxwell
