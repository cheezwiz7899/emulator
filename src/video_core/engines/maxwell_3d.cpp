// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/common_funcs.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/draw_manager.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"

extern "C" {
volatile unsigned long long citron_uh_watch_gpu_addr = 0;
volatile unsigned long long citron_uh_watch_cpu_addr = 0;
volatile unsigned long long citron_uh_watch_host_addr = 0;
}

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

namespace {
bool UltrahandTraceEnabled() {
    static const bool enabled = std::getenv("CITRON_ULTRAHAND_TRACE") != nullptr;
    return enabled;
}

bool UltrahandPassTraceEnabled() {
    static const bool enabled = std::getenv("CITRON_UH_PASS_TRACE") != nullptr;
    return enabled;
}

bool IsUltrahandConditionCpuRange(VAddr addr, u64 size) {
    constexpr VAddr condition_begin = 0x1B200000;
    constexpr VAddr condition_end = 0x1B220000;
    return addr < condition_end && addr + size > condition_begin;
}

bool UltrahandBreakProbeEnabled(GPUVAddr address) {
    static const bool enabled = std::getenv("CITRON_UH_BREAK_ON_COND_DIFF") != nullptr;
    if (!enabled) {
        return false;
    }
    const char* const arm_file = std::getenv("CITRON_UH_BREAK_ARM_FILE");
    if (arm_file != nullptr && arm_file[0] != '\0' && !std::filesystem::exists(arm_file)) {
        return false;
    }
    static bool already_broke = false;
    if (already_broke) {
        return false;
    }
    const char* const target = std::getenv("CITRON_UH_BREAK_GPU_ADDR");
    if (target != nullptr && target[0] != '\0') {
        const u64 target_addr = std::strtoull(target, nullptr, 0);
        if (target_addr != address) {
            return false;
        }
    }
    return true;
}

bool UltrahandBreakOnCondDiff(GPUVAddr address) {
    if (!UltrahandBreakProbeEnabled(address)) {
        return false;
    }
    static bool already_broke = false;
    if (already_broke) {
        return false;
    }
    already_broke = true;
    return true;
}
} // namespace

Maxwell3D::Maxwell3D(Core::System& system_, MemoryManager& memory_manager_)
    : draw_manager{std::make_unique<DrawManager>(this)}
    , system{system_}
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++) {
        execution_mask[i] = IsMethodExecutable(static_cast<u32>(i));
    }
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_params.insert(macro_params.end(), base_start, base_start + amount);
    for (size_t i = 0; i < amount; i++)
        macro_addresses.push_back(current_dma_segment + i * sizeof(u32));
    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink();
        CallMacroMethod(executing_macro, macro_params);
        macro_params.clear();
        macro_addresses.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelNormal()) {
        // Skip parameter refresh for Low accuracy - ultimate performance
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = std::max(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
        num_vertices = std::max(
            num_vertices, address_size / std::max(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    static constexpr std::array<size_t, 3> max_sizes = {std::numeric_limits<u8>::max(), std::numeric_limits<u16>::max(), std::numeric_limits<u32>::max()};
    const size_t byte_size = regs.index_buffer.FormatSizeInBytes();
    const size_t log2_byte_size = Common::Log2Ceil64(byte_size);
    const size_t cap{GetMaxCurrentVertices() * 4 * byte_size};
    const size_t lower_cap = std::min<size_t>(size_t(end_address - start_address), cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_sizes[log2_byte_size]) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_state.reg_array[method] = argument;
        return argument;
    }
    if (control == Regs::ShadowRamControl::Replay) {
        return shadow_state.reg_array[method];
    }
    return argument;
}

void Maxwell3D::ConsumeSinkImpl() {
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            ProcessDirtyRegisters(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    regs.reg_array[method] = argument;
    for (const auto& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = Regs::ShadowRamControl(nonshadow_argument);
        return;
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        return macro_engine.ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        return macro_engine.AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager->ProcessMethodCall(method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(u32 method, const std::vector<u32>& parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(macro_positions.size());

    // Execute the current macro.
    macro_engine.Execute(*this, macro_positions[entry], parameters);

    draw_manager->DrawDeferred();
}

void Maxwell3D::CallMethod(u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(method, &method_argument, 1, is_last_call);
        return;
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");

    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::CallMultiMethod(u32 method, const u32* base_start, u32 amount,
                                u32 methods_pending) {
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(method, base_start, amount, amount == methods_pending);
        return;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    macro_engine.AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    if (regs.load_mme.start_address_ptr < macro_positions.size()) [[likely]] {
        macro_positions[regs.load_mme.start_address_ptr++] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    const bool render_enable_report =
        regs.render_enable_override == Regs::RenderEnable::Override::UseRenderEnable &&
        sequence_address == regs.render_enable.Address() &&
        (regs.render_enable.mode == Regs::RenderEnable::Mode::IfEqual ||
         regs.render_enable.mode == Regs::RenderEnable::Mode::IfNotEqual) &&
        query_type == VideoCommon::QueryType::Payload;
    if (render_enable_report) {
        flags |= VideoCommon::QueryPropertiesFlags::IsRenderEnableReport;
    }
    if (UltrahandTraceEnabled()) {
        LOG_WARNING(HW_GPU,
                 "UHTRACE query addr=0x{:016X} type={} op={} raw=0x{:08X} "
                 "dword={} flags=0x{:X} short={} "
                 "execute={} render_report={} render_addr=0x{:016X} render_mode={} render_override={} "
                 "payload=0x{:08X} subreport={}",
                 sequence_address, static_cast<u32>(query_type),
                 static_cast<u32>(regs.report_semaphore.query.operation.Value()),
                 regs.report_semaphore.query.raw, regs.report_semaphore.query.dword_number.Value(),
                 static_cast<u32>(flags), regs.report_semaphore.query.short_query != 0,
                 execute_on, render_enable_report, regs.render_enable.Address(),
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), payload, subreport);
    }
    const auto sequence_cpu_addr = memory_manager.GpuToCpuAddress(sequence_address);
    const auto render_cpu_addr = memory_manager.GpuToCpuAddress(regs.render_enable.Address());
    const bool pass_query =
        sequence_cpu_addr &&
        IsUltrahandConditionCpuRange(*sequence_cpu_addr,
                                     regs.report_semaphore.query.short_query != 0 ? sizeof(u32)
                                                                                  : sizeof(u64));
    const bool pass_render =
        render_cpu_addr && IsUltrahandConditionCpuRange(*render_cpu_addr,
                                                        sizeof(Regs::ReportSemaphore::Compare));
    if (UltrahandPassTraceEnabled() && (pass_query || pass_render)) {
        const auto& draw_state = draw_manager->GetDrawState();
        const auto& rt0 = regs.rt[0];
        const auto& rt1 = regs.rt[1];
        const auto& zeta = regs.zeta;
        const auto shader_offset = [this](Regs::ShaderType type) {
            const auto index = static_cast<std::size_t>(type);
            return regs.IsShaderConfigEnabled(index) ? regs.pipelines[index].offset : 0U;
        };
        const u32 draw_count =
            draw_state.draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
        LOG_WARNING(HW_GPU,
                    "UHTRACE pass_query gpu=0x{:016X} cpu=0x{:016X} type={} op={} "
                    "flags=0x{:X} short={} execute={} render_report={} "
                    "render_gpu=0x{:016X} render_cpu=0x{:016X} render_mode={} "
                    "render_override={} payload=0x{:08X} subreport={} "
                    "ctx topology={} indexed={} count={} inst={} zpass={} "
                    "rt0=0x{:016X}/{}x{}/fmt{} rt1=0x{:016X}/{}x{}/fmt{} "
                    "zeta=0x{:016X}/fmt{} clip={}x{} mrt=0x{:08X} "
                    "sh_vb=0x{:08X} sh_fs=0x{:08X} sh_gs=0x{:08X}",
                    sequence_address, static_cast<u64>(sequence_cpu_addr.value_or(0)),
                    static_cast<u32>(query_type),
                    static_cast<u32>(regs.report_semaphore.query.operation.Value()),
                    static_cast<u32>(flags), regs.report_semaphore.query.short_query != 0,
                    execute_on, render_enable_report, regs.render_enable.Address(),
                    static_cast<u64>(render_cpu_addr.value_or(0)),
                    static_cast<u32>(regs.render_enable.mode),
                    static_cast<u32>(regs.render_enable_override), payload, subreport,
                    static_cast<u32>(draw_state.topology), draw_state.draw_indexed, draw_count,
                    draw_state.instance_count, regs.zpass_pixel_count_enable != 0, rt0.Address(),
                    rt0.width, rt0.height, static_cast<u32>(rt0.format), rt1.Address(),
                    rt1.width, rt1.height, static_cast<u32>(rt1.format), zeta.Address(),
                    static_cast<u32>(zeta.format), static_cast<u32>(regs.surface_clip.width),
                    static_cast<u32>(regs.surface_clip.height), regs.color_target_mrt_enable,
                    shader_offset(Regs::ShaderType::VertexB),
                    shader_offset(Regs::ShaderType::Pixel),
                    shader_offset(Regs::ShaderType::Geometry));
    }
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    const GPUVAddr condition_address{regs.render_enable.Address()};
    const bool trace = UltrahandTraceEnabled();
    const bool break_probe = UltrahandBreakProbeEnabled(condition_address);
    const bool pass_trace = UltrahandPassTraceEnabled();
    const auto condition_cpu_addr = memory_manager.GpuToCpuAddress(condition_address);
    const bool pass_condition =
        condition_cpu_addr &&
        IsUltrahandConditionCpuRange(*condition_cpu_addr,
                                     sizeof(Regs::ReportSemaphore::Compare));
    const bool accelerated = rasterizer->AccelerateConditionalRendering();
    const auto read_compare = [&](const char* mode_name) {
        Regs::ReportSemaphore::Compare before{};
        Regs::ReportSemaphore::Compare cmp{};
        Regs::ReportSemaphore::Compare after{};
        const auto cpu_addr = memory_manager.GpuToCpuAddress(condition_address);
        if (trace || break_probe) {
            memory_manager.ReadBlockUnsafe(condition_address, &before, sizeof(before));
        }
        memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
        if (trace || break_probe) {
            memory_manager.ReadBlockUnsafe(condition_address, &after, sizeof(after));
            if (std::memcmp(&before, &cmp, sizeof(cmp)) != 0 ||
                std::memcmp(&cmp, &after, sizeof(cmp)) != 0) {
                const auto& draw_state = draw_manager->GetDrawState();
                const auto& rt0 = regs.rt[0];
                const auto& rt1 = regs.rt[1];
                const auto& zeta = regs.zeta;
                const auto shader_offset = [this](Regs::ShaderType type) {
                    const auto index = static_cast<std::size_t>(type);
                    return regs.IsShaderConfigEnabled(index) ? regs.pipelines[index].offset : 0U;
                };
                const u32 draw_count =
                    draw_state.draw_indexed ? draw_state.index_buffer.count
                                            : draw_state.vertex_buffer.count;
                const auto* const host_ptr = memory_manager.GetPointer(condition_address);
                const u64 host_addr =
                    static_cast<u64>(reinterpret_cast<std::uintptr_t>(host_ptr));
                if (trace) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE cond_read_diff mode={} addr=0x{:016X} "
                                "before=0x{:08X}/0x{:08X},0x{:08X}/0x{:08X} "
                                "safe=0x{:08X}/0x{:08X},0x{:08X}/0x{:08X} "
                                "after=0x{:08X}/0x{:08X},0x{:08X}/0x{:08X} "
                                "ctx topology={} indexed={} count={} inst={} zpass={} "
                                "rt0=0x{:016X}/{}x{}/fmt{} rt1=0x{:016X}/{}x{}/fmt{} "
                                "zeta=0x{:016X}/fmt{} clip={}x{} mrt=0x{:08X} "
                                "sh_vb=0x{:08X} sh_fs=0x{:08X} sh_gs=0x{:08X} "
                                "cpu=0x{:016X} host=0x{:016X}",
                                mode_name, condition_address, before.initial_sequence,
                                before.initial_mode, before.current_sequence, before.current_mode,
                                cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence,
                                cmp.current_mode, after.initial_sequence, after.initial_mode,
                                after.current_sequence, after.current_mode,
                                static_cast<u32>(draw_state.topology), draw_state.draw_indexed,
                                draw_count, draw_state.instance_count,
                                regs.zpass_pixel_count_enable != 0, rt0.Address(), rt0.width,
                                rt0.height, static_cast<u32>(rt0.format), rt1.Address(), rt1.width,
                                rt1.height, static_cast<u32>(rt1.format), zeta.Address(),
                                static_cast<u32>(zeta.format),
                                static_cast<u32>(regs.surface_clip.width),
                                static_cast<u32>(regs.surface_clip.height),
                                regs.color_target_mrt_enable,
                                shader_offset(Regs::ShaderType::VertexB),
                                shader_offset(Regs::ShaderType::Pixel),
                                shader_offset(Regs::ShaderType::Geometry),
                                static_cast<u64>(cpu_addr.value_or(0)), host_addr);
                }
                if (UltrahandBreakOnCondDiff(condition_address)) {
                    citron_uh_watch_gpu_addr = condition_address;
                    citron_uh_watch_cpu_addr = static_cast<u64>(cpu_addr.value_or(0));
                    citron_uh_watch_host_addr = host_addr;
                    LOG_WARNING(HW_GPU,
                                "UHTRACE cond_debug_break addr=0x{:016X} cpu=0x{:016X} "
                                "host=0x{:016X}",
                                condition_address, static_cast<u64>(cpu_addr.value_or(0)),
                                host_addr);
                    Crash();
                }
            }
        }
        return cmp;
    };
    const auto log_pass_condition = [&](const char* mode_name,
                                        const Regs::ReportSemaphore::Compare* cmp, bool execute,
                                        bool hcr_accelerated) {
        if (!pass_trace || !pass_condition) {
            return;
        }
        const auto& draw_state = draw_manager->GetDrawState();
        const auto& rt0 = regs.rt[0];
        const auto& rt1 = regs.rt[1];
        const auto& zeta = regs.zeta;
        const auto shader_offset = [this](Regs::ShaderType type) {
            const auto index = static_cast<std::size_t>(type);
            return regs.IsShaderConfigEnabled(index) ? regs.pipelines[index].offset : 0U;
        };
        const u32 draw_count =
            draw_state.draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
        const Regs::ReportSemaphore::Compare empty{};
        const auto& values = cmp != nullptr ? *cmp : empty;
        LOG_WARNING(HW_GPU,
                    "UHTRACE pass_cond mode={} addr=0x{:016X} cpu=0x{:016X} "
                    "accelerated={} execute={} has_cmp={} "
                    "init=0x{:08X}/0x{:08X} cur=0x{:08X}/0x{:08X} "
                    "ctx topology={} indexed={} count={} inst={} zpass={} "
                    "rt0=0x{:016X}/{}x{}/fmt{} rt1=0x{:016X}/{}x{}/fmt{} "
                    "zeta=0x{:016X}/fmt{} clip={}x{} mrt=0x{:08X} "
                    "sh_vb=0x{:08X} sh_fs=0x{:08X} sh_gs=0x{:08X}",
                    mode_name, condition_address,
                    static_cast<u64>(condition_cpu_addr.value_or(0)), hcr_accelerated, execute,
                    cmp != nullptr, values.initial_sequence, values.initial_mode,
                    values.current_sequence, values.current_mode,
                    static_cast<u32>(draw_state.topology), draw_state.draw_indexed, draw_count,
                    draw_state.instance_count, regs.zpass_pixel_count_enable != 0, rt0.Address(),
                    rt0.width, rt0.height, static_cast<u32>(rt0.format), rt1.Address(),
                    rt1.width, rt1.height, static_cast<u32>(rt1.format), zeta.Address(),
                    static_cast<u32>(zeta.format), static_cast<u32>(regs.surface_clip.width),
                    static_cast<u32>(regs.surface_clip.height), regs.color_target_mrt_enable,
                    shader_offset(Regs::ShaderType::VertexB),
                    shader_offset(Regs::ShaderType::Pixel),
                    shader_offset(Regs::ShaderType::Geometry));
    };
    if (accelerated) {
        execute_on = true;
        log_pass_condition("accelerated", nullptr, execute_on, true);
        if (trace) {
            LOG_WARNING(HW_GPU,
                 "UHTRACE cond accelerated=1 addr=0x{:016X} override={} mode={} execute=1",
                     condition_address, static_cast<u32>(regs.render_enable_override),
                     static_cast<u32>(regs.render_enable.mode));
        }
        return;
    }
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        log_pass_condition("always", nullptr, execute_on, false);
        if (trace) {
            LOG_WARNING(HW_GPU,
                 "UHTRACE cond override addr=0x{:016X} override={} mode={} execute=1",
                     condition_address, static_cast<u32>(regs.render_enable_override),
                     static_cast<u32>(regs.render_enable.mode));
        }
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        log_pass_condition("never", nullptr, execute_on, false);
        if (trace) {
            LOG_WARNING(HW_GPU,
                 "UHTRACE cond override addr=0x{:016X} override={} mode={} execute=0",
                     condition_address, static_cast<u32>(regs.render_enable_override),
                     static_cast<u32>(regs.render_enable.mode));
        }
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            log_pass_condition("true", nullptr, execute_on, false);
            if (trace) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE cond mode_true addr=0x{:016X} override={} mode={} execute=1",
                         condition_address, static_cast<u32>(regs.render_enable_override),
                         static_cast<u32>(regs.render_enable.mode));
            }
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            log_pass_condition("false", nullptr, execute_on, false);
            if (trace) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE cond mode_false addr=0x{:016X} override={} mode={} execute=0",
                         condition_address, static_cast<u32>(regs.render_enable_override),
                         static_cast<u32>(regs.render_enable.mode));
            }
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            Regs::ReportSemaphore::Compare cmp = read_compare("conditional");
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            log_pass_condition("conditional", &cmp, execute_on, false);
            if (trace) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE cond conditional addr=0x{:016X} init=0x{:08X}/0x{:08X} "
                         "cur=0x{:08X}/0x{:08X} execute={}",
                         condition_address, cmp.initial_sequence, cmp.initial_mode,
                         cmp.current_sequence, cmp.current_mode, execute_on);
            }
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            Regs::ReportSemaphore::Compare cmp = read_compare("equal");
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            log_pass_condition("equal", &cmp, execute_on, false);
            if (trace) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE cond equal addr=0x{:016X} init=0x{:08X}/0x{:08X} "
                         "cur=0x{:08X}/0x{:08X} execute={}",
                         condition_address, cmp.initial_sequence, cmp.initial_mode,
                         cmp.current_sequence, cmp.current_mode, execute_on);
            }
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            Regs::ReportSemaphore::Compare cmp = read_compare("not_equal");
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            log_pass_condition("not_equal", &cmp, execute_on, false);
            if (trace) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE cond notequal addr=0x{:016X} init=0x{:08X}/0x{:08X} "
                         "cur=0x{:08X}/0x{:08X} execute={}",
                         condition_address, cmp.initial_sequence, cmp.initial_mode,
                         cmp.current_sequence, cmp.current_mode, execute_on);
            }
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    ASSERT(regs.const_buffer.offset <= regs.const_buffer.size);

    const GPUVAddr address{buffer_address + regs.const_buffer.offset};
    const size_t copy_size = amount * sizeof(u32);
    memory_manager.WriteBlockCached(address, start_base, copy_size);

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT_MSG(method < Regs::NUM_REGS, "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

} // namespace Tegra::Engines
