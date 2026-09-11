// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate/translate.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader {
struct HostTranslateInfo;
}

namespace Shader::Maxwell {

// Builds the initial frontend IR. This is not an immutable scanner boundary:
// BuildASL still reads instructions, SPH/stage/local-memory state, and some
// instruction paths consult cbuf-backed Environment state. The returned
// Program owns handles allocated from the supplied pools, so it is neither
// independently owned nor safe to persist. FinalizeProgramTemplate() defers
// further environment-dependent lowering, but does not make this first half
// state-independent.
[[nodiscard]] IR::Program BuildProgramTemplate(ObjectPool<IR::Inst>& inst_pool,
                                                ObjectPool<IR::Block>& block_pool,
                                                Environment& env, Flow::CFG& cfg,
                                                const HostTranslateInfo& host_info,
                                                std::span<const PredecodedInstruction> predecoded = {});

// Applies all real-environment-dependent lowering to a template. Keeping this
// as a separate entry point makes the old TranslateProgram sequence explicit
// without changing it; callers that do not retain templates should continue
// using TranslateProgram().
void FinalizeProgramTemplate(IR::Program& program, Environment& env,
                             const HostTranslateInfo& host_info);

[[nodiscard]] IR::Program TranslateProgram(ObjectPool<IR::Inst>& inst_pool,
                                           ObjectPool<IR::Block>& block_pool, Environment& env,
                                           Flow::CFG& cfg, const HostTranslateInfo& host_info);

[[nodiscard]] IR::Program MergeDualVertexPrograms(IR::Program& vertex_a, IR::Program& vertex_b,
                                                  Environment& env_vertex_b);

void ConvertLegacyToGeneric(IR::Program& program, const RuntimeInfo& runtime_info);

// Maxwell v1 and older Nvidia cards don't support setting gl_Layer from non-geometry stages.
// This creates a workaround by setting the layer as a generic output and creating a
// passthrough geometry shader that reads the generic and sets the layer.
[[nodiscard]] IR::Program GenerateGeometryPassthrough(ObjectPool<IR::Inst>& inst_pool,
                                                      ObjectPool<IR::Block>& block_pool,
                                                      const HostTranslateInfo& host_info,
                                                      IR::Program& source_program,
                                                      Shader::OutputTopology output_topology);

} // namespace Shader::Maxwell
