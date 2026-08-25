// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <optional>
#include <unordered_set>

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Optimization {
namespace {
struct ConstBufferAddr {
    u32 index;
    u32 offset;
    u32 shift_left;
    u32 secondary_index;
    u32 secondary_offset;
    u32 secondary_shift_left;
    IR::U32 dynamic_offset;
    u32 count;
    bool has_secondary;
};

struct TextureInst {
    ConstBufferAddr cbuf;
    IR::Inst* inst;
    IR::Block* block;
};

using TextureInstVector = boost::container::small_vector<TextureInst, 24>;

constexpr u32 DESCRIPTOR_SIZE = 8;
constexpr u32 DESCRIPTOR_SIZE_SHIFT = static_cast<u32>(std::countr_zero(DESCRIPTOR_SIZE));

constexpr u32 BINDLESS_ARRAY_LENGTH = 1024;
constexpr u32 BINDLESS_FALLBACK_LENGTH = BINDLESS_ARRAY_LENGTH;

u32 BindlessCountForCbuf(Environment& env, u32 cbuf_index) {
    const u32 cbuf_size = env.ReadCbufSize(cbuf_index);
    if (cbuf_size == 0) {
        return BINDLESS_FALLBACK_LENGTH;
    }
    return std::min(std::max(cbuf_size / DESCRIPTOR_SIZE, 1u), BINDLESS_ARRAY_LENGTH);
}

IR::Opcode IndexedInstruction(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::BindlessImageSampleImplicitLod:
    case IR::Opcode::BoundImageSampleImplicitLod:
        return IR::Opcode::ImageSampleImplicitLod;
    case IR::Opcode::BoundImageSampleExplicitLod:
    case IR::Opcode::BindlessImageSampleExplicitLod:
        return IR::Opcode::ImageSampleExplicitLod;
    case IR::Opcode::BoundImageSampleDrefImplicitLod:
    case IR::Opcode::BindlessImageSampleDrefImplicitLod:
        return IR::Opcode::ImageSampleDrefImplicitLod;
    case IR::Opcode::BoundImageSampleDrefExplicitLod:
    case IR::Opcode::BindlessImageSampleDrefExplicitLod:
        return IR::Opcode::ImageSampleDrefExplicitLod;
    case IR::Opcode::BindlessImageGather:
    case IR::Opcode::BoundImageGather:
        return IR::Opcode::ImageGather;
    case IR::Opcode::BindlessImageGatherDref:
    case IR::Opcode::BoundImageGatherDref:
        return IR::Opcode::ImageGatherDref;
    case IR::Opcode::BindlessImageFetch:
    case IR::Opcode::BoundImageFetch:
        return IR::Opcode::ImageFetch;
    case IR::Opcode::BoundImageQueryDimensions:
    case IR::Opcode::BindlessImageQueryDimensions:
        return IR::Opcode::ImageQueryDimensions;
    case IR::Opcode::BoundImageQueryLod:
    case IR::Opcode::BindlessImageQueryLod:
        return IR::Opcode::ImageQueryLod;
    case IR::Opcode::BoundImageGradient:
    case IR::Opcode::BindlessImageGradient:
        return IR::Opcode::ImageGradient;
    case IR::Opcode::BoundImageRead:
    case IR::Opcode::BindlessImageRead:
        return IR::Opcode::ImageRead;
    case IR::Opcode::BoundImageWrite:
    case IR::Opcode::BindlessImageWrite:
        return IR::Opcode::ImageWrite;
    case IR::Opcode::BoundImageAtomicIAdd32:
    case IR::Opcode::BindlessImageAtomicIAdd32:
        return IR::Opcode::ImageAtomicIAdd32;
    case IR::Opcode::BoundImageAtomicSMin32:
    case IR::Opcode::BindlessImageAtomicSMin32:
        return IR::Opcode::ImageAtomicSMin32;
    case IR::Opcode::BoundImageAtomicUMin32:
    case IR::Opcode::BindlessImageAtomicUMin32:
        return IR::Opcode::ImageAtomicUMin32;
    case IR::Opcode::BoundImageAtomicSMax32:
    case IR::Opcode::BindlessImageAtomicSMax32:
        return IR::Opcode::ImageAtomicSMax32;
    case IR::Opcode::BoundImageAtomicUMax32:
    case IR::Opcode::BindlessImageAtomicUMax32:
        return IR::Opcode::ImageAtomicUMax32;
    case IR::Opcode::BoundImageAtomicInc32:
    case IR::Opcode::BindlessImageAtomicInc32:
        return IR::Opcode::ImageAtomicInc32;
    case IR::Opcode::BoundImageAtomicDec32:
    case IR::Opcode::BindlessImageAtomicDec32:
        return IR::Opcode::ImageAtomicDec32;
    case IR::Opcode::BoundImageAtomicAnd32:
    case IR::Opcode::BindlessImageAtomicAnd32:
        return IR::Opcode::ImageAtomicAnd32;
    case IR::Opcode::BoundImageAtomicOr32:
    case IR::Opcode::BindlessImageAtomicOr32:
        return IR::Opcode::ImageAtomicOr32;
    case IR::Opcode::BoundImageAtomicXor32:
    case IR::Opcode::BindlessImageAtomicXor32:
        return IR::Opcode::ImageAtomicXor32;
    case IR::Opcode::BoundImageAtomicExchange32:
    case IR::Opcode::BindlessImageAtomicExchange32:
        return IR::Opcode::ImageAtomicExchange32;
    default:
        return IR::Opcode::Void;
    }
}

bool IsBindless(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::BindlessImageSampleImplicitLod:
    case IR::Opcode::BindlessImageSampleExplicitLod:
    case IR::Opcode::BindlessImageSampleDrefImplicitLod:
    case IR::Opcode::BindlessImageSampleDrefExplicitLod:
    case IR::Opcode::BindlessImageGather:
    case IR::Opcode::BindlessImageGatherDref:
    case IR::Opcode::BindlessImageFetch:
    case IR::Opcode::BindlessImageQueryDimensions:
    case IR::Opcode::BindlessImageQueryLod:
    case IR::Opcode::BindlessImageGradient:
    case IR::Opcode::BindlessImageRead:
    case IR::Opcode::BindlessImageWrite:
    case IR::Opcode::BindlessImageAtomicIAdd32:
    case IR::Opcode::BindlessImageAtomicSMin32:
    case IR::Opcode::BindlessImageAtomicUMin32:
    case IR::Opcode::BindlessImageAtomicSMax32:
    case IR::Opcode::BindlessImageAtomicUMax32:
    case IR::Opcode::BindlessImageAtomicInc32:
    case IR::Opcode::BindlessImageAtomicDec32:
    case IR::Opcode::BindlessImageAtomicAnd32:
    case IR::Opcode::BindlessImageAtomicOr32:
    case IR::Opcode::BindlessImageAtomicXor32:
    case IR::Opcode::BindlessImageAtomicExchange32:
        return true;
    case IR::Opcode::BoundImageSampleImplicitLod:
    case IR::Opcode::BoundImageSampleExplicitLod:
    case IR::Opcode::BoundImageSampleDrefImplicitLod:
    case IR::Opcode::BoundImageSampleDrefExplicitLod:
    case IR::Opcode::BoundImageGather:
    case IR::Opcode::BoundImageGatherDref:
    case IR::Opcode::BoundImageFetch:
    case IR::Opcode::BoundImageQueryDimensions:
    case IR::Opcode::BoundImageQueryLod:
    case IR::Opcode::BoundImageGradient:
    case IR::Opcode::BoundImageRead:
    case IR::Opcode::BoundImageWrite:
    case IR::Opcode::BoundImageAtomicIAdd32:
    case IR::Opcode::BoundImageAtomicSMin32:
    case IR::Opcode::BoundImageAtomicUMin32:
    case IR::Opcode::BoundImageAtomicSMax32:
    case IR::Opcode::BoundImageAtomicUMax32:
    case IR::Opcode::BoundImageAtomicInc32:
    case IR::Opcode::BoundImageAtomicDec32:
    case IR::Opcode::BoundImageAtomicAnd32:
    case IR::Opcode::BoundImageAtomicOr32:
    case IR::Opcode::BoundImageAtomicXor32:
    case IR::Opcode::BoundImageAtomicExchange32:
        return false;
    default:
        throw InvalidArgument("Invalid opcode {}", inst.GetOpcode());
    }
}

bool IsTextureInstruction(const IR::Inst& inst) {
    return IndexedInstruction(inst) != IR::Opcode::Void;
}

std::optional<ConstBufferAddr> TryGetConstBuffer(const IR::Inst* inst, Environment& env);

std::optional<ConstBufferAddr> Track(const IR::Value& value, Environment& env) {
    return IR::BreadthFirstSearch(
        value, [&env](const IR::Inst* inst) { return TryGetConstBuffer(inst, env); });
}

std::optional<u32> TryGetConstant(IR::Value& value, Environment& env) {
    const IR::Inst* inst = value.InstRecursive();
    if (inst->GetOpcode() != IR::Opcode::GetCbufU32) {
        return std::nullopt;
    }
    const IR::Value index{inst->Arg(0)};
    const IR::Value offset{inst->Arg(1)};
    if (!index.IsImmediate()) {
        return std::nullopt;
    }
    if (!offset.IsImmediate()) {
        return std::nullopt;
    }
    const auto index_number = index.U32();
    if (index_number != 1) {
        return std::nullopt;
    }
    const auto offset_number = offset.U32();
    return env.ReadCbufValue(index_number, offset_number);
}

std::optional<ConstBufferAddr> TryGetConstBuffer(const IR::Inst* inst, Environment& env) {
    switch (inst->GetOpcode()) {
    default:
        return std::nullopt;
    case IR::Opcode::BitwiseOr32: {
        std::optional lhs{Track(inst->Arg(0), env)};
        std::optional rhs{Track(inst->Arg(1), env)};
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        if (lhs->has_secondary || rhs->has_secondary) {
            return std::nullopt;
        }
        if (lhs->count > 1 || rhs->count > 1) {
            return std::nullopt;
        }
        if (lhs->shift_left > 0 || lhs->index > rhs->index || lhs->offset > rhs->offset) {
            std::swap(lhs, rhs);
        }
        return ConstBufferAddr{
            .index = lhs->index,
            .offset = lhs->offset,
            .shift_left = lhs->shift_left,
            .secondary_index = rhs->index,
            .secondary_offset = rhs->offset,
            .secondary_shift_left = rhs->shift_left,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = true,
        };
    }
    case IR::Opcode::ShiftLeftLogical32: {
        const IR::Value shift{inst->Arg(1)};
        if (!shift.IsImmediate()) {
            return std::nullopt;
        }
        std::optional lhs{Track(inst->Arg(0), env)};
        if (lhs) {
            lhs->shift_left = shift.U32();
        }
        return lhs;
        break;
    }
    case IR::Opcode::BitwiseAnd32: {
        IR::Value op1{inst->Arg(0)};
        IR::Value op2{inst->Arg(1)};
        if (op1.IsImmediate()) {
            std::swap(op1, op2);
        }
        if (!op2.IsImmediate() && !op1.IsImmediate()) {
            do {
                auto try_index = TryGetConstant(op1, env);
                if (try_index) {
                    op1 = op2;
                    op2 = IR::Value{*try_index};
                    break;
                }
                auto try_index_2 = TryGetConstant(op2, env);
                if (try_index_2) {
                    op2 = IR::Value{*try_index_2};
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        std::optional lhs{Track(op1, env)};
        if (lhs) {
            lhs->shift_left = static_cast<u32>(std::countr_zero(op2.U32()));
        }
        return lhs;
        break;
    }
    case IR::Opcode::GetCbufU32x2:
    case IR::Opcode::GetCbufU32:
        break;
    }
    const IR::Value index{inst->Arg(0)};
    const IR::Value offset{inst->Arg(1)};
    if (!index.IsImmediate()) {
        // Reading a bindless texture from variable indices is valid
        // but not supported here at the moment
        return std::nullopt;
    }
    if (offset.IsImmediate()) {
        return ConstBufferAddr{
            .index = index.U32(),
            .offset = offset.U32(),
            .shift_left = 0,
            .secondary_index = 0,
            .secondary_offset = 0,
            .secondary_shift_left = 0,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = false,
        };
    }
    IR::Inst* const offset_inst{offset.InstRecursive()};
    if (offset_inst->GetOpcode() != IR::Opcode::IAdd32) {
        return std::nullopt;
    }
    u32 base_offset{};
    IR::U32 dynamic_offset;
    if (offset_inst->Arg(0).IsImmediate()) {
        base_offset = offset_inst->Arg(0).U32();
        dynamic_offset = IR::U32{offset_inst->Arg(1)};
    } else if (offset_inst->Arg(1).IsImmediate()) {
        base_offset = offset_inst->Arg(1).U32();
        dynamic_offset = IR::U32{offset_inst->Arg(0)};
    } else {
        return std::nullopt;
    }
    return ConstBufferAddr{
        .index = index.U32(),
        .offset = base_offset,
        .shift_left = 0,
        .secondary_index = 0,
        .secondary_offset = 0,
        .secondary_shift_left = 0,
        .dynamic_offset = dynamic_offset,
        .count = BindlessCountForCbuf(env, index.U32()),
        .has_secondary = false,
    };
}

TextureInst MakeInst(Environment& env, IR::Block* block, IR::Inst& inst) {
    ConstBufferAddr addr;
    if (IsBindless(inst)) {
        const std::optional<ConstBufferAddr> track_addr{Track(inst.Arg(0), env)};
        if (!track_addr) {
            // Enhanced bindless texture handling for UE4 games like Hogwarts Legacy
            // Instead of throwing an exception, we'll use a fallback approach
            LOG_WARNING(Shader, "Failed to track bindless texture constant buffer, using fallback");

            // Use a default constant buffer address as fallback
            addr = ConstBufferAddr{
                .index = env.TextureBoundBuffer(),
                .offset = 0,
                .shift_left = 0,
                .secondary_index = 0,
                .secondary_offset = 0,
                .secondary_shift_left = 0,
                .dynamic_offset = {},
                .count = 1,
                .has_secondary = false,
            };
        } else {
            addr = *track_addr;
        }
    } else {
        addr = ConstBufferAddr{
            .index = env.TextureBoundBuffer(),
            .offset = inst.Arg(0).U32(),
            .shift_left = 0,
            .secondary_index = 0,
            .secondary_offset = 0,
            .secondary_shift_left = 0,
            .dynamic_offset = {},
            .count = 1,
            .has_secondary = false,
        };
    }
    return TextureInst{
        .cbuf = addr,
        .inst = &inst,
        .block = block,
    };
}

u32 GetTextureHandle(Environment& env, const ConstBufferAddr& cbuf) {
    const u32 secondary_index{cbuf.has_secondary ? cbuf.secondary_index : cbuf.index};
    const u32 secondary_offset{cbuf.has_secondary ? cbuf.secondary_offset : cbuf.offset};
    // ReadCbufValueForTextureHandle, not ReadCbufValue: this is the ONE place in the
    // whole codebase that resolves a bindless texture handle from a cbuf read (see its
    // doc comment in environment.h) — tagging it here, at the only call site that
    // actually knows that's what it's doing, is what lets a later diagnostic ask "how
    // much of cbuf_key's entropy is actually just re-deriving what texture_key already
    // captures on its own?" without guessing at intent after the fact.
    const u32 lhs_raw{env.ReadCbufValueForTextureHandle(cbuf.index, cbuf.offset) << cbuf.shift_left};
    const u32 rhs_raw{env.ReadCbufValueForTextureHandle(secondary_index, secondary_offset)
                      << cbuf.secondary_shift_left};
    return lhs_raw | rhs_raw;
}

TextureType ReadTextureType(Environment& env, const ConstBufferAddr& cbuf) {
    const u32 handle{GetTextureHandle(env, cbuf)};
    const TextureType type{env.ReadTextureType(handle)};
    // See RecordResolvedTextureType's doc comment in environment.h — Phase 4 feasibility
    // instrumentation, no-op for every Environment except GenericEnvironment. Passing handle
    // too now (not just cbuf coordinates) — needed by the texture_key exclusion fix, see
    // CapturedPhase4PrototypeHandles's doc comment in shader_environment.h.
    env.RecordResolvedTextureType(cbuf.index, cbuf.offset, handle, type);
    return type;
}

TexturePixelFormat ReadTexturePixelFormat(Environment& env, const ConstBufferAddr& cbuf) {
    const TexturePixelFormat format{env.ReadTexturePixelFormat(GetTextureHandle(env, cbuf))};
    // See RecordResolvedTexturePixelFormat's doc comment in environment.h — same Phase 4
    // instrumentation as ReadTextureType above.
    env.RecordResolvedTexturePixelFormat(cbuf.index, cbuf.offset, format);
    return format;
}

bool IsTexturePixelFormatInteger(Environment& env, const ConstBufferAddr& cbuf) {
    const bool is_integer{env.IsTexturePixelFormatInteger(GetTextureHandle(env, cbuf))};
    // Third axis of the same instrumentation as ReadTextureType/ReadTexturePixelFormat above
    // -- see RecordResolvedIsTexturePixelFormatInteger's doc comment in environment.h.
    env.RecordResolvedIsTexturePixelFormatInteger(cbuf.index, cbuf.offset, is_integer);
    return is_integer;
}

class Descriptors {
public:
    explicit Descriptors(TextureBufferDescriptors& texture_buffer_descriptors_,
                         ImageBufferDescriptors& image_buffer_descriptors_,
                         TextureDescriptors& texture_descriptors_,
                         ImageDescriptors& image_descriptors_)
        : texture_buffer_descriptors{texture_buffer_descriptors_},
          image_buffer_descriptors{image_buffer_descriptors_},
          texture_descriptors{texture_descriptors_}, image_descriptors{image_descriptors_} {}

    u32 Add(const TextureBufferDescriptor& desc) {
        return Add(texture_buffer_descriptors, desc, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset &&
                   desc.shift_left == existing.shift_left &&
                   desc.secondary_cbuf_index == existing.secondary_cbuf_index &&
                   desc.secondary_cbuf_offset == existing.secondary_cbuf_offset &&
                   desc.secondary_shift_left == existing.secondary_shift_left &&
                   desc.count == existing.count && desc.size_shift == existing.size_shift &&
                   desc.has_secondary == existing.has_secondary;
        });
    }

    u32 Add(const ImageBufferDescriptor& desc) {
        const u32 index{Add(image_buffer_descriptors, desc, [&desc](const auto& existing) {
            return desc.format == existing.format && desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.size_shift == existing.size_shift;
        })};
        image_buffer_descriptors[index].is_written |= desc.is_written;
        image_buffer_descriptors[index].is_read |= desc.is_read;
        image_buffer_descriptors[index].is_integer |= desc.is_integer;
        return index;
    }

    u32 Add(const TextureDescriptor& desc) {
        const u32 index{Add(texture_descriptors, desc, [&desc](const auto& existing) {
            return desc.type == existing.type && desc.is_depth == existing.is_depth &&
                   desc.has_secondary == existing.has_secondary &&
                   desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset &&
                   desc.shift_left == existing.shift_left &&
                   desc.secondary_cbuf_index == existing.secondary_cbuf_index &&
                   desc.secondary_cbuf_offset == existing.secondary_cbuf_offset &&
                   desc.secondary_shift_left == existing.secondary_shift_left &&
                   desc.count == existing.count && desc.size_shift == existing.size_shift;
        })};
        // TODO: Read this from TIC
        texture_descriptors[index].is_multisample |= desc.is_multisample;
        return index;
    }

    u32 Add(const ImageDescriptor& desc) {
        const u32 index{Add(image_descriptors, desc, [&desc](const auto& existing) {
            return desc.type == existing.type && desc.format == existing.format &&
                   desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.size_shift == existing.size_shift;
        })};
        image_descriptors[index].is_written |= desc.is_written;
        image_descriptors[index].is_read |= desc.is_read;
        image_descriptors[index].is_integer |= desc.is_integer;
        return index;
    }

private:
    template <typename Descriptors, typename Descriptor, typename Func>
    static u32 Add(Descriptors& descriptors, const Descriptor& desc, Func&& pred) {
        // TODO: Handle arrays
        const auto it{std::ranges::find_if(descriptors, pred)};
        if (it != descriptors.end()) {
            return static_cast<u32>(std::distance(descriptors.begin(), it));
        }
        descriptors.push_back(desc);
        return static_cast<u32>(descriptors.size()) - 1;
    }

    TextureBufferDescriptors& texture_buffer_descriptors;
    ImageBufferDescriptors& image_buffer_descriptors;
    TextureDescriptors& texture_descriptors;
    ImageDescriptors& image_descriptors;
};

void PatchImageSampleImplicitLod(IR::Block& block, IR::Inst& inst) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const IR::Value coord(inst.Arg(1));
    const IR::Value handle(ir.Imm32(0));
    const IR::U32 lod{ir.Imm32(0)};
    const IR::U1 skip_mips{ir.Imm1(true)};
    const IR::Value texture_size = ir.ImageQueryDimension(handle, lod, skip_mips, info);
    inst.SetArg(
        1, ir.CompositeConstruct(
               ir.FPMul(IR::F32(ir.CompositeExtract(coord, 0)),
                        ir.FPRecip(ir.ConvertUToF(32, 32, ir.CompositeExtract(texture_size, 0)))),
               ir.FPMul(IR::F32(ir.CompositeExtract(coord, 1)),
                        ir.FPRecip(ir.ConvertUToF(32, 32, ir.CompositeExtract(texture_size, 1))))));
}

bool IsPixelFormatSNorm(TexturePixelFormat pixel_format) {
    switch (pixel_format) {
    case TexturePixelFormat::A8B8G8R8_SNORM:
    case TexturePixelFormat::R8G8_SNORM:
    case TexturePixelFormat::R8_SNORM:
    case TexturePixelFormat::R16G16B16A16_SNORM:
    case TexturePixelFormat::R16G16_SNORM:
    case TexturePixelFormat::R16_SNORM:
        return true;
    default:
        return false;
    }
}

void PatchTexelFetch(IR::Block& block, IR::Inst& inst, TexturePixelFormat pixel_format) {
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    auto get_max_value = [pixel_format]() -> float {
        switch (pixel_format) {
        case TexturePixelFormat::A8B8G8R8_SNORM:
        case TexturePixelFormat::R8G8_SNORM:
        case TexturePixelFormat::R8_SNORM:
            return 1.f / std::numeric_limits<char>::max();
        case TexturePixelFormat::R16G16B16A16_SNORM:
        case TexturePixelFormat::R16G16_SNORM:
        case TexturePixelFormat::R16_SNORM:
            return 1.f / std::numeric_limits<short>::max();
        default:
            throw InvalidArgument("Invalid texture pixel format");
        }
    };

    const IR::Value new_inst{&*block.PrependNewInst(it, inst)};
    const IR::F32 x(ir.CompositeExtract(new_inst, 0));
    const IR::F32 y(ir.CompositeExtract(new_inst, 1));
    const IR::F32 z(ir.CompositeExtract(new_inst, 2));
    const IR::F32 w(ir.CompositeExtract(new_inst, 3));
    const IR::F16F32F64 max_value(ir.Imm32(get_max_value()));
    const IR::Value converted =
        ir.CompositeConstruct(ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(x)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(y)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(z)), max_value),
                              ir.FPMul(ir.ConvertSToF(32, 32, ir.BitCast<IR::U32>(w)), max_value));
    inst.ReplaceUsesWith(converted);
}
} // Anonymous namespace

void TexturePass(Environment& env, IR::Program& program, const HostTranslateInfo& host_info) {
    TextureInstVector to_replace;
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsTextureInstruction(inst)) {
                continue;
            }
            to_replace.push_back(MakeInst(env, block, inst));
        }
    }
    // Sort instructions to visit textures by constant buffer index, then by offset
    std::ranges::sort(to_replace, [](const auto& lhs, const auto& rhs) {
        return lhs.cbuf.offset < rhs.cbuf.offset;
    });
    std::stable_sort(to_replace.begin(), to_replace.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.cbuf.index < rhs.cbuf.index;
    });

    // Phase 4 correctness scoping. The branch/OpPhi that makes this mechanism actually
    // correct exists ONLY for ImageQueryDimensions (EmitImageQueryDimensions,
    // emit_spirv_image.cpp) -- every other texture instruction (Sample, Fetch, Gather, Read,
    // Write, Gradient, QueryLod) still unconditionally reads through the canonical/primary
    // descriptor via Texture()/TextureImage() (emit_spirv_image.cpp), with no branch at all.
    // Marking a slot phase4_prototype_polymorphic when it's ALSO referenced by one of those
    // other instructions -- which, since descriptors.Add() below merges same-slot references
    // into one shared TextureDescriptor, is a real structural risk, not a hypothetical one --
    // would silently sample/fetch/gather through the wrong-typed descriptor whenever the real
    // content is the array variant. Not observed in any real session across TotK, SMO, or any
    // other title tested so far, but not proven absent either, and correctly fixing Sample/
    // Fetch/Gather for the array case needs a synthesized array-layer coordinate that
    // Color2D-canonicalized IR was never taught to compute -- a materially bigger, less
    // certain change than declining to apply this optimization where it can't yet be made
    // correct. Scoped out here instead: any coordinate used by a non-ImageQueryDimensions
    // texture instruction anywhere in this shader is excluded from phase4 treatment for this
    // shader specifically -- falls back to the exact ordinary (pre-Phase-4) per-draw
    // re-translation behavior for that one shader, same as an inactive coordinate would, not
    // a new code path of its own.
    // Second, independent scoping condition, found the same way as the first (reading the
    // real call sites rather than assuming): PipelineCache::ResolvePhase4PrototypeSpecValue
    // (vk_pipeline_cache.cpp) only ever resolves a slot's real value from the FRAGMENT stage's
    // cbuf state -- shader_stages[kFragmentSoftwareIndex], hardcoded, no other stage read at
    // all. MakePipeline (vk_graphics_pipeline.cpp) attaches that one resolved value's
    // VkSpecializationInfo to whichever stage(s) declared a matching SpecId, without checking
    // WHICH stage that is -- so a polymorphic descriptor in a non-fragment graphics stage
    // (vertex, geometry, tessellation) would receive a value resolved from the FRAGMENT
    // shader's cbuf state at the same slot number, not its own, which is only correct by
    // coincidence if both stages happen to bind the same texture at the same slot. A compute
    // shader is worse: ComputePipelineCacheKey carries no phase4 field at all, and
    // CreateComputePipeline never calls ResolvePhase4PrototypeSpecValue, so a polymorphic
    // compute descriptor's spec constant would keep SPIR-V's own default (OpSpecConstantFalse
    // -- always the canonical branch) unconditionally. Same fix shape as above: fragment-only,
    // whole-program, checked once rather than per instruction since it's a property of the
    // program, not of any one texture reference in it.
    const bool is_phase4_safe_stage{program.stage == Shader::Stage::Fragment};

    std::unordered_set<u64> phase4_unsafe_coords;
    for (const TextureInst& ti : to_replace) {
        if (ti.inst->GetOpcode() != IR::Opcode::ImageQueryDimensions) {
            phase4_unsafe_coords.insert((u64{ti.cbuf.index} << 32) | u64{ti.cbuf.offset});
        }
    }
    // Two more scoping conditions, found by asking the same question a third and fourth time
    // rather than assuming the first two were the whole list: does anything downstream of this
    // marking depend on an assumption this site never actually checks?
    //
    // cbuf.count != 1: Phase4PrototypeBindingCount (shader_info.h) returns 1 or 2 -- how many
    // EXTRA bindings polymorphism needs -- entirely independent of desc.count, which is how
    // many array ELEMENTS this one binding already has (a genuine runtime-indexed texture
    // array, texture[idx], not the Color2D/ColorArray2D dimensionality this mechanism exists
    // for). ResolvePhase4PrototypeSpecValue (vk_pipeline_cache.cpp) resolves exactly one handle
    // at the base cbuf offset and applies that one spec-constant value uniformly -- if count>1
    // ever means N independently-varying array elements, one resolved value can't be right for
    // all of them. Whether DefineTextures' emission side happens to still be safe for this case
    // was not traced all the way through; scoped out instead of trusting that chain, the same
    // choice made for instruction coverage and stage safety above rather than proving it by
    // more reading.
    //
    // cbuf.has_secondary: not a new finding -- ResolvePhase4PrototypeSpecValue's OWN doc
    // comment (vk_pipeline_cache.cpp) already named this as an unconfirmed assumption
    // ("Assumes no secondary cbuf combine for any known slot... If it turns out this slot does
    // use a secondary combine, this function silently resolves the wrong handle") and left it
    // unconfirmed rather than unsafe-by-construction. GetTextureHandle (texture_pass.cpp) can
    // OR together two separate cbuf reads via has_secondary/secondary_cbuf_index/
    // secondary_cbuf_offset when a descriptor needs it; ResolvePhase4PrototypeSpecValue only
    // ever reads the primary cbuf_index/cbuf_offset. Closing the gap the same way as the other
    // three rather than leaving it as a named-but-open risk.
    const auto safe_phase4_slot_id = [&](const ConstBufferAddr& cbuf) -> std::optional<u32> {
        if (!is_phase4_safe_stage) {
            return std::nullopt;
        }
        if (phase4_unsafe_coords.contains((u64{cbuf.index} << 32) | u64{cbuf.offset})) {
            return std::nullopt;
        }
        if (cbuf.count != 1) {
            return std::nullopt;
        }
        if (cbuf.has_secondary) {
            return std::nullopt;
        }
        return Shader::Phase4PrototypeSlotId(cbuf.index, cbuf.offset);
    };

    Descriptors descriptors{
        program.info.texture_buffer_descriptors,
        program.info.image_buffer_descriptors,
        program.info.texture_descriptors,
        program.info.image_descriptors,
    };
    for (TextureInst& texture_inst : to_replace) {
        // TODO: Handle arrays
        IR::Inst* const inst{texture_inst.inst};
        inst->ReplaceOpcode(IndexedInstruction(*inst));

        const auto& cbuf{texture_inst.cbuf};
        auto flags{inst->Flags<IR::TextureInstInfo>()};
        bool is_multisample{false};
        switch (inst->GetOpcode()) {
        case IR::Opcode::ImageQueryDimensions: {
            const TextureType resolved{ReadTextureType(env, cbuf)};
            // Phase 4 (specialization-constant texture-type resolution). Keep calling
            // ReadTextureType() above regardless (so the existing distinct-value-tracking
            // instrumentation still sees what this draw actually resolved to -- that data
            // stays useful, and is how new slots get discovered, even for a slot this
            // canonicalizes), but for an ACTUALLY safe-to-polymorphize slot (see
            // safe_phase4_slot_id above -- active, fragment-stage, AND not also touched by a
            // non-ImageQueryDimensions instruction in this shader), DON'T bake the real
            // per-draw resolution into flags.type the normal way. Every translation of this slot --
            // whichever type this particular draw resolved to -- assigns the SAME canonical
            // value instead, so every translation produces an identical TextureDescriptor
            // (same .type, same everything), which is what makes them dedupe to the one
            // shared, polymorphic SPIR-V module instead of fragmenting into two cache entries
            // the way an un-prototyped slot would.
            constexpr TextureType kPrototypeCanonicalType = TextureType::Color2D;
            const bool would_be_active{Shader::IsPhase4PrototypeSlot(cbuf.index, cbuf.offset)};
            const bool is_safe{safe_phase4_slot_id(cbuf).has_value()};
            if (would_be_active && !is_safe) {
                // Diagnostic only, not throttled: real data so far suggests this should be
                // rare (order of a handful of known-slot events per session, not per-draw) --
                // if this fires often in practice, that itself is worth knowing rather than
                // hiding behind a throttle. Confirms directly, from a real session, whether
                // any currently-known coordinate is actually affected by any of the four
                // scoping conditions rather than leaving it as this session's untested guess.
                // Four separate messages, not one shared with a reason parameter, specifically
                // so grepping a log for one exclusion reason doesn't also match another.
                if (!is_phase4_safe_stage) {
                    LOG_WARNING(Shader,
                                "Phase 4: cbuf_index={} cbuf_offset={} is an active slot but "
                                "this program's stage is not Fragment -- declining to "
                                "polymorphize it here (falling back to ordinary per-draw "
                                "re-translation) since only the fragment stage's cbuf state "
                                "is ever resolved for this mechanism",
                                cbuf.index, cbuf.offset);
                }
                if (phase4_unsafe_coords.contains((u64{cbuf.index} << 32) | u64{cbuf.offset})) {
                    LOG_WARNING(Shader,
                                "Phase 4: cbuf_index={} cbuf_offset={} is an active slot but is "
                                "also referenced by a non-ImageQueryDimensions instruction in "
                                "this shader -- declining to polymorphize it here (falling back "
                                "to ordinary per-draw re-translation) since only "
                                "ImageQueryDimensions has a correct branch for it",
                                cbuf.index, cbuf.offset);
                }
                if (cbuf.count != 1) {
                    LOG_WARNING(Shader,
                                "Phase 4: cbuf_index={} cbuf_offset={} is an active slot but "
                                "count={} (not a single descriptor) -- declining to "
                                "polymorphize it here (falling back to ordinary per-draw "
                                "re-translation) since only a single resolved value is ever "
                                "computed, not one per array element",
                                cbuf.index, cbuf.offset, cbuf.count);
                }
                if (cbuf.has_secondary) {
                    LOG_WARNING(Shader,
                                "Phase 4: cbuf_index={} cbuf_offset={} is an active slot but "
                                "has a secondary cbuf combine -- declining to polymorphize it "
                                "here (falling back to ordinary per-draw re-translation) since "
                                "only the primary cbuf read is ever resolved, not the combined "
                                "handle",
                                cbuf.index, cbuf.offset);
                }
            }
            if (is_safe) {
                flags.type.Assign(kPrototypeCanonicalType);
            } else {
                flags.type.Assign(resolved);
            }
            inst->SetFlags(flags);
            break;
        }
        case IR::Opcode::ImageSampleImplicitLod:
            if (flags.type != TextureType::Color2D) {
                break;
            }
            if (ReadTextureType(env, cbuf) == TextureType::Color2DRect) {
                PatchImageSampleImplicitLod(*texture_inst.block, *texture_inst.inst);
            }
            break;
        case IR::Opcode::ImageFetch:
            if (flags.type == TextureType::Color2D || flags.type == TextureType::Color2DRect ||
                flags.type == TextureType::ColorArray2D) {
                is_multisample = !inst->Arg(4).IsEmpty();
            } else {
                inst->SetArg(4, IR::U32{});
            }
            if (flags.type != TextureType::Color1D) {
                break;
            }
            if (ReadTextureType(env, cbuf) == TextureType::Buffer) {
                // Replace with the bound texture type only when it's a texture buffer
                // If the instruction is 1D and the bound type is 2D, don't change the code and let
                // the rasterizer robustness handle it
                // This happens on Fire Emblem: Three Houses
                flags.type.Assign(TextureType::Buffer);
            }
            break;
        default:
            break;
        }
        u32 index;
        switch (inst->GetOpcode()) {
        case IR::Opcode::ImageRead:
        case IR::Opcode::ImageAtomicIAdd32:
        case IR::Opcode::ImageAtomicSMin32:
        case IR::Opcode::ImageAtomicUMin32:
        case IR::Opcode::ImageAtomicSMax32:
        case IR::Opcode::ImageAtomicUMax32:
        case IR::Opcode::ImageAtomicInc32:
        case IR::Opcode::ImageAtomicDec32:
        case IR::Opcode::ImageAtomicAnd32:
        case IR::Opcode::ImageAtomicOr32:
        case IR::Opcode::ImageAtomicXor32:
        case IR::Opcode::ImageAtomicExchange32:
        case IR::Opcode::ImageWrite: {
            if (cbuf.has_secondary) {
                throw NotImplementedException("Unexpected separate sampler");
            }
            const bool is_written{inst->GetOpcode() != IR::Opcode::ImageRead};
            const bool is_read{inst->GetOpcode() != IR::Opcode::ImageWrite};
            const bool is_integer{IsTexturePixelFormatInteger(env, cbuf)};
            if (flags.type == TextureType::Buffer) {
                index = descriptors.Add(ImageBufferDescriptor{
                    .format = flags.image_format,
                    .is_written = is_written,
                    .is_read = is_read,
                    .is_integer = is_integer,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .count = cbuf.count,
                    .size_shift = DESCRIPTOR_SIZE_SHIFT,
                });
            } else {
                index = descriptors.Add(ImageDescriptor{
                    .type = flags.type,
                    .format = flags.image_format,
                    .is_written = is_written,
                    .is_read = is_read,
                    .is_integer = is_integer,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .count = cbuf.count,
                    .size_shift = DESCRIPTOR_SIZE_SHIFT,
                });
            }
            break;
        }
        default:
            if (flags.type == TextureType::Buffer) {
                index = descriptors.Add(TextureBufferDescriptor{
                    .has_secondary = cbuf.has_secondary,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .shift_left = cbuf.shift_left,
                    .secondary_cbuf_index = cbuf.secondary_index,
                    .secondary_cbuf_offset = cbuf.secondary_offset,
                    .secondary_shift_left = cbuf.secondary_shift_left,
                    .count = cbuf.count,
                    .size_shift = DESCRIPTOR_SIZE_SHIFT,
                });
            } else {
                // Computed once and reused for both fields below rather than calling
                // safe_phase4_slot_id(cbuf) again, so this site can't end up with a
                // polymorphic flag and slot_id that were independently looked up and (in
                // principle, if the table changed between the two calls) disagreed. Uses the
                // safety-scoped predicate (see its doc comment above, near
                // phase4_unsafe_coords) rather than calling Shader::Phase4PrototypeSlotId
                // directly, so this descriptor's flag can never disagree with whether
                // ImageQueryDimensions' own case (above) actually canonicalized flags.type
                // for it -- getting those two out of sync would be worse than doing nothing:
                // a descriptor marked polymorphic whose type was never canonicalized, or
                // vice versa.
                const std::optional<u32> phase4_slot_id{safe_phase4_slot_id(cbuf)};
                index = descriptors.Add(TextureDescriptor{
                    .type = flags.type,
                    .is_depth = flags.is_depth != 0,
                    .is_multisample = is_multisample,
                    .has_secondary = cbuf.has_secondary,
                    .cbuf_index = cbuf.index,
                    .cbuf_offset = cbuf.offset,
                    .shift_left = cbuf.shift_left,
                    .secondary_cbuf_index = cbuf.secondary_index,
                    .secondary_cbuf_offset = cbuf.secondary_offset,
                    .secondary_shift_left = cbuf.secondary_shift_left,
                    .count = cbuf.count,
                    .size_shift = DESCRIPTOR_SIZE_SHIFT,
                    .phase4_prototype_polymorphic = phase4_slot_id.has_value(),
                    .phase4_prototype_slot_id = phase4_slot_id.value_or(0),
                });
            }
            break;
        }
        flags.descriptor_index.Assign(index);
        inst->SetFlags(flags);

        if (cbuf.count > 1) {
            const auto insert_point{IR::Block::InstructionList::s_iterator_to(*inst)};
            IR::IREmitter ir{*texture_inst.block, insert_point};
            const IR::U32 shift{ir.Imm32(std::countr_zero(DESCRIPTOR_SIZE))};
            inst->SetArg(0, ir.UMin(ir.ShiftRightArithmetic(cbuf.dynamic_offset, shift),
                                    ir.Imm32(cbuf.count - 1)));
        } else {
            inst->SetArg(0, IR::Value{});
        }

        if (!host_info.support_snorm_render_buffer && inst->GetOpcode() == IR::Opcode::ImageFetch &&
            flags.type == TextureType::Buffer) {
            const auto pixel_format = ReadTexturePixelFormat(env, cbuf);
            if (IsPixelFormatSNorm(pixel_format)) {
                PatchTexelFetch(*texture_inst.block, *texture_inst.inst, pixel_format);
            }
        }
    }
}

void JoinTextureInfo(Info& base, Info& source) {
    Descriptors descriptors{
        base.texture_buffer_descriptors,
        base.image_buffer_descriptors,
        base.texture_descriptors,
        base.image_descriptors,
    };
    for (auto& desc : source.texture_buffer_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.image_buffer_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.texture_descriptors) {
        descriptors.Add(desc);
    }
    for (auto& desc : source.image_descriptors) {
        descriptors.Add(desc);
    }
}

} // namespace Shader::Optimization
