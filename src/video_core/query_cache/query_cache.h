// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/assert.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/query_cache/bank_base.h"
#include "video_core/query_cache/query_base.h"
#include "video_core/query_cache/query_cache_base.h"
#include "video_core/query_cache/query_stream.h"
#include "video_core/query_cache/types.h"

namespace VideoCommon {

inline bool UltrahandTraceEnabled() {
    static const bool enabled = std::getenv("CITRON_ULTRAHAND_TRACE") != nullptr;
    return enabled;
}

inline bool UltrahandPassTraceEnabled() {
    static const bool enabled = std::getenv("CITRON_UH_PASS_TRACE") != nullptr;
    return enabled;
}

inline bool IsUltrahandConditionCpuRange(VAddr addr, u64 size) {
    constexpr VAddr condition_begin = 0x1B200000;
    constexpr VAddr condition_end = 0x1B220000;
    return addr < condition_end && addr + size > condition_begin;
}

inline bool IsUltrahandConditionGpuRange(GPUVAddr addr, u64 size) {
    constexpr GPUVAddr condition_begin = 0x0000000503200000ULL;
    constexpr GPUVAddr condition_end = 0x0000000503400000ULL;
    return addr < condition_end && addr + size > condition_begin;
}

inline bool IsStaleGuestQueryWrite(const QueryBase& query, const void* pointer, size_t size,
                                   u64* current_value = nullptr) {
    if (pointer == nullptr || True(query.flags & QueryFlagBits::IsHostManaged)) {
        return false;
    }
    u64 current{};
    std::memcpy(&current, pointer, size);
    if (current_value != nullptr) {
        *current_value = current;
    }
    const u64 mask = size >= sizeof(u64) ? ~0ULL : ((1ULL << (size * 8)) - 1ULL);
    if ((current & mask) != (query.guest_snapshot & mask)) {
        return true;
    }
    return false;
}

inline bool IsUnsafeRenderEnableGuestWrite([[maybe_unused]] const QueryBase& query,
                                           [[maybe_unused]] u64 write_value,
                                           [[maybe_unused]] u64 current_value) {
    return false;
}

struct SyncValuesStruct {
    VAddr address;
    u64 value;
    u64 size;

    static constexpr bool GeneratesBaseBuffer = true;
};

template <typename Traits>
class GuestStreamer : public SimpleStreamer<GuestQuery> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    GuestStreamer(size_t id_, RuntimeType& runtime_)
        : SimpleStreamer<GuestQuery>(id_), runtime{runtime_} {}

    virtual ~GuestStreamer() = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        auto new_id = BuildQuery(has_timestamp, address, static_cast<u64>(value));
        pending_sync.push_back(new_id);
        return new_id;
    }

    bool HasPendingSync() const override {
        return !pending_sync.empty();
    }

    void SyncWrites() override {
        if (pending_sync.empty()) {
            return;
        }
        std::vector<SyncValuesStruct> sync_values;
        sync_values.reserve(pending_sync.size());
        for (size_t pending_id : pending_sync) {
            auto& query = slot_queries[pending_id];
            if (True(query.flags & QueryFlagBits::IsRewritten) ||
                True(query.flags & QueryFlagBits::IsInvalidated)) {
                continue;
            }
            query.flags |= QueryFlagBits::IsHostSynced;
            sync_values.emplace_back(SyncValuesStruct{
                .address = query.guest_address,
                .value = query.value,
                .size = static_cast<u64>(True(query.flags & QueryFlagBits::HasTimestamp) ? 8 : 4)});
        }
        pending_sync.clear();
        if (sync_values.size() > 0) {
            runtime.template SyncValues<SyncValuesStruct>(sync_values);
        }
    }

private:
    RuntimeType& runtime;
    std::deque<size_t> pending_sync;
};

template <typename Traits>
class StubStreamer : public GuestStreamer<Traits> {
public:
    using RuntimeType = typename Traits::RuntimeType;

    StubStreamer(size_t id_, RuntimeType& runtime_, u32 stub_value_)
        : GuestStreamer<Traits>(id_, runtime_), stub_value{stub_value_} {}

    ~StubStreamer() override = default;

    size_t WriteCounter(VAddr address, bool has_timestamp, [[maybe_unused]] u32 value,
                        std::optional<u32> subreport = std::nullopt) override {
        size_t new_id =
            GuestStreamer<Traits>::WriteCounter(address, has_timestamp, stub_value, subreport);
        return new_id;
    }

private:
    u32 stub_value;
};

template <typename Traits>
struct QueryCacheBase<Traits>::QueryCacheBaseImpl {
    using RuntimeType = typename Traits::RuntimeType;

    QueryCacheBaseImpl(QueryCacheBase<Traits>* owner_, VideoCore::RasterizerInterface& rasterizer_,
                       Tegra::MaxwellDeviceMemoryManager& device_memory_, RuntimeType& runtime_,
                       Tegra::GPU& gpu_)
        : owner{owner_}, rasterizer{rasterizer_},
          device_memory{device_memory_}, runtime{runtime_}, gpu{gpu_} {
        streamer_mask = 0;
        for (size_t i = 0; i < static_cast<size_t>(QueryType::MaxQueryTypes); i++) {
            streamers[i] = runtime.GetStreamerInterface(static_cast<QueryType>(i));
            if (streamers[i]) {
                streamer_mask |= 1ULL << streamers[i]->GetId();
            }
        }
    }

    template <typename Func>
    void ForEachStreamerIn(u64 mask, Func&& func) {
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result<Func, StreamerInterface*>, bool>;
        while (mask != 0) {
            size_t position = std::countr_zero(mask);
            mask &= ~(1ULL << position);
            if constexpr (RETURNS_BOOL) {
                if (func(streamers[position])) {
                    return;
                }
            } else {
                func(streamers[position]);
            }
        }
    }

    template <typename Func>
    void ForEachStreamer(Func&& func) {
        ForEachStreamerIn(streamer_mask, func);
    }

    QueryBase* ObtainQuery(QueryCacheBase<Traits>::QueryLocation location) {
        size_t which_stream = location.stream_id.Value();
        auto* streamer = streamers[which_stream];
        if (!streamer) {
            return nullptr;
        }
        return streamer->GetQuery(location.query_id.Value());
    }

    void TrackConditionPage(DAddr address) {
        if (!IsUltrahandConditionCpuRange(address, QueryCacheBase<Traits>::QUERY_REPORT_SIZE)) {
            return;
        }
        const DAddr page = address & ~static_cast<DAddr>(Core::DEVICE_PAGEMASK);
        if (tracked_condition_pages.insert(page).second) {
            device_memory.PinPagesCached(page, Core::DEVICE_PAGESIZE);
            if (UltrahandTraceEnabled()) {
                LOG_WARNING(HW_GPU, "UHTRACE qc_track_page cpu=0x{:016X}", page);
            }
        }
    }

    QueryCacheBase<Traits>* owner;
    VideoCore::RasterizerInterface& rasterizer;
    Tegra::MaxwellDeviceMemoryManager& device_memory;
    RuntimeType& runtime;
    Tegra::GPU& gpu;
    std::array<StreamerInterface*, static_cast<size_t>(QueryType::MaxQueryTypes)> streamers;
    u64 streamer_mask;
    std::mutex flush_guard;
    std::deque<u64> flushes_pending;
    std::vector<QueryCacheBase<Traits>::QueryLocation> pending_unregister;
    std::unordered_set<GPUVAddr> render_enable_compare_addresses;
    std::unordered_set<DAddr> tracked_condition_pages;
};

template <typename Traits>
QueryCacheBase<Traits>::QueryCacheBase(Tegra::GPU& gpu_,
                                       VideoCore::RasterizerInterface& rasterizer_,
                                       Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                       RuntimeType& runtime_)
    : cached_queries{} {
    impl = std::make_unique<QueryCacheBase<Traits>::QueryCacheBaseImpl>(
        this, rasterizer_, device_memory_, runtime_, gpu_);
}

template <typename Traits>
QueryCacheBase<Traits>::~QueryCacheBase() = default;

template <typename Traits>
void QueryCacheBase<Traits>::CounterEnable(QueryType counter_type, bool is_enabled) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    if (is_enabled) {
        streamer->StartCounter();
    } else {
        streamer->PauseCounter();
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterClose(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNREACHABLE();
        return;
    }
    streamer->CloseCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReset(QueryType counter_type) {
    size_t index = static_cast<size_t>(counter_type);
    StreamerInterface* streamer = impl->streamers[index];
    if (!streamer) [[unlikely]] {
        UNIMPLEMENTED();
        return;
    }
    streamer->ResetCounter();
}

template <typename Traits>
void QueryCacheBase<Traits>::BindToChannel(s32 id) {
    VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo>::BindToChannel(id);
    impl->runtime.Bind3DEngine(maxwell3d);
}

template <typename Traits>
void QueryCacheBase<Traits>::CounterReport(GPUVAddr addr, QueryType counter_type,
                                           QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    const bool has_timestamp = True(flags & QueryPropertiesFlags::HasTimeout);
    const bool is_fence = True(flags & QueryPropertiesFlags::IsAFence);
    const GPUVAddr original_addr = addr;
    size_t streamer_id = static_cast<size_t>(counter_type);
    auto* streamer = impl->streamers[streamer_id];
    if (streamer == nullptr) [[unlikely]] {
        counter_type = QueryType::Payload;
        payload = 1U;
        streamer_id = static_cast<size_t>(counter_type);
        streamer = impl->streamers[streamer_id];
    }
    auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(addr);
    if (!cpu_addr_opt) [[unlikely]] {
        return;
    }
    DAddr cpu_addr = *cpu_addr_opt;
    impl->TrackConditionPage(cpu_addr);
    const size_t new_query_id = streamer->WriteCounter(cpu_addr, has_timestamp, payload, subreport);
    auto* query = streamer->GetQuery(new_query_id);
    if (True(flags & QueryPropertiesFlags::IsRenderEnableReport)) {
        query->flags |= QueryFlagBits::IsRenderEnableReport;
    }
    if (UltrahandTraceEnabled()) {
        LOG_WARNING(HW_GPU,
                 "UHTRACE qc_report gpu=0x{:016X} report_gpu=0x{:016X} cpu=0x{:016X} "
                 "type={} streamer={} id={} "
                 "fence={} timestamp={} payload=0x{:08X} subreport={}",
                 original_addr, addr, cpu_addr, static_cast<u32>(counter_type), streamer_id,
                 new_query_id, is_fence, has_timestamp, payload, subreport);
    }
    if (is_fence) {
        query->flags |= QueryFlagBits::IsFence;
    }
    QueryLocation query_location{};
    query_location.stream_id.Assign(static_cast<u32>(streamer_id));
    query_location.query_id.Assign(static_cast<u32>(new_query_id));
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    u8* pointer = impl->device_memory.template GetPointer<u8>(cpu_addr);
    u8* pointer_timestamp = impl->device_memory.template GetPointer<u8>(cpu_addr + 8);
    const size_t snapshot_size = has_timestamp ? sizeof(query->value) : sizeof(payload);
    if (pointer) {
        std::memcpy(&query->guest_snapshot, pointer, snapshot_size);
    }
    if (UltrahandPassTraceEnabled() &&
        (IsUltrahandConditionCpuRange(cpu_addr, snapshot_size) ||
         IsUltrahandConditionGpuRange(original_addr, snapshot_size))) {
        LOG_WARNING(HW_GPU,
                    "UHTRACE pass_report gpu=0x{:016X} cpu=0x{:016X} type={} streamer={} "
                    "id={} flags=0x{:X} fence={} timestamp={} payload=0x{:08X} "
                    "subreport={} snapshot=0x{:016X} value=0x{:016X}",
                    original_addr, cpu_addr, static_cast<u32>(counter_type), streamer_id,
                    new_query_id, static_cast<u32>(query->flags), is_fence, has_timestamp, payload,
                    subreport, query->guest_snapshot, query->value);
    }
    bool is_synced = !Settings::IsGPULevelNormal() && is_fence;
    std::function<void()> operation([this, is_synced, streamer, query_base = query, query_location,
                                     pointer, pointer_timestamp] {
        if (True(query_base->flags & QueryFlagBits::IsGuestSynced)) {
            if (!is_synced) [[likely]] {
                impl->pending_unregister.push_back(query_location);
            }
            return;
        }
        if (True(query_base->flags & QueryFlagBits::IsInvalidated) ||
            True(query_base->flags & QueryFlagBits::IsRewritten)) {
            if (!is_synced) [[likely]] {
                impl->pending_unregister.push_back(query_location);
            }
            return;
        }
        if (False(query_base->flags & QueryFlagBits::IsFinalValueSynced)) [[unlikely]] {
            ASSERT(false);
            return;
        }
        query_base->value += streamer->GetAmendValue();
        streamer->SetAccumulationValue(query_base->value);
        if (True(query_base->flags & QueryFlagBits::HasTimestamp)) {
            u64 timestamp = impl->gpu.GetTicks();
            u64 current{};
            if (IsStaleGuestQueryWrite(*query_base, pointer, sizeof(query_base->value),
                                       &current)) {
                if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                    IsUltrahandConditionCpuRange(query_base->guest_address,
                                                 sizeof(query_base->value))) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE qc_deferred_guest_skip_stale cpu=0x{:016X} size=8 "
                                "flags=0x{:X} value=0x{:016X} snapshot=0x{:016X} "
                                "current=0x{:016X}",
                                query_base->guest_address, static_cast<u32>(query_base->flags),
                                query_base->value, query_base->guest_snapshot, current);
                }
                query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
                if (!is_synced) [[likely]] {
                    impl->pending_unregister.push_back(query_location);
                }
                return;
            }
            if (IsUnsafeRenderEnableGuestWrite(*query_base, query_base->value, current)) {
                if (UltrahandTraceEnabled()) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE qc_deferred_guest_skip_render_enable cpu=0x{:016X} "
                                "size=8 flags=0x{:X} value=0x{:016X} snapshot=0x{:016X} "
                                "current=0x{:016X}",
                                query_base->guest_address, static_cast<u32>(query_base->flags),
                                query_base->value, query_base->guest_snapshot, current);
                }
                query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
                if (!is_synced) [[likely]] {
                    impl->pending_unregister.push_back(query_location);
                }
                return;
            }
            if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(query_base->value))) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE qc_deferred_guest_write cpu=0x{:016X} size=8 "
                            "flags=0x{:X} value=0x{:016X} timestamp=1",
                            query_base->guest_address, static_cast<u32>(query_base->flags),
                            query_base->value);
            }
            if (pointer_timestamp) std::memcpy(pointer_timestamp, &timestamp, sizeof(timestamp));
            if (pointer) std::memcpy(pointer, &query_base->value, sizeof(query_base->value));
        } else {
            u32 value = static_cast<u32>(query_base->value);
            u64 current{};
            if (IsStaleGuestQueryWrite(*query_base, pointer, sizeof(value), &current)) {
                if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                    IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(value))) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE qc_deferred_guest_skip_stale cpu=0x{:016X} size=4 "
                                "flags=0x{:X} value=0x{:08X} snapshot=0x{:016X} "
                                "current=0x{:016X}",
                                query_base->guest_address, static_cast<u32>(query_base->flags),
                                value, query_base->guest_snapshot, current);
                }
                query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
                if (!is_synced) [[likely]] {
                    impl->pending_unregister.push_back(query_location);
                }
                return;
            }
            if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(value))) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE qc_deferred_guest_write cpu=0x{:016X} size=4 "
                            "flags=0x{:X} value=0x{:08X} timestamp=0",
                            query_base->guest_address, static_cast<u32>(query_base->flags), value);
            }
            if (pointer) std::memcpy(pointer, &value, sizeof(value));
        }
        query_base->flags |= QueryFlagBits::IsGuestSynced;
        if (!is_synced) [[likely]] {
            impl->pending_unregister.push_back(query_location);
        }
    });
    if (is_fence) {
        impl->rasterizer.SignalFence(std::move(operation));
    } else {
        if (!Settings::IsGPULevelNormal() && counter_type == QueryType::Payload) {
            // Low accuracy: Immediately write payload for ultimate performance
            if (has_timestamp) {
                u64 timestamp = impl->gpu.GetTicks();
                u64 value = static_cast<u64>(payload);
                if (pointer_timestamp) std::memcpy(pointer_timestamp, &timestamp, sizeof(timestamp));
                if (pointer) std::memcpy(pointer, &value, sizeof(value));
            } else {
                if (pointer) std::memcpy(pointer, &payload, sizeof(payload));
            }
            streamer->Free(new_query_id);
            return;
        }
        impl->rasterizer.SyncOperation(std::move(operation));
    }
    if (is_synced) {
        streamer->Free(new_query_id);
        return;
    }
    auto [cont_addr, base] = gen_caching_indexing(cpu_addr);
    {
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.try_emplace(cont_addr);
        auto& sub_container = it1.first->second;
        auto it_current = sub_container.find(base);
        if (it_current == sub_container.end()) {
            sub_container.insert_or_assign(base, query_location);
            return;
        }
        auto* old_query = impl->ObtainQuery(it_current->second);
        old_query->flags |= QueryFlagBits::IsRewritten;
        sub_container.insert_or_assign(base, query_location);
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::UnregisterPending() {
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    std::scoped_lock lock(cache_mutex);
    for (QueryLocation loc : impl->pending_unregister) {
        const auto [streamer_id, query_id] = loc.unpack();
        auto* streamer = impl->streamers[streamer_id];
        if (!streamer) [[unlikely]] {
            continue;
        }
        auto* query = streamer->GetQuery(query_id);
        if (!query) [[unlikely]] {
            continue;
        }
        auto [cont_addr, base] = gen_caching_indexing(query->guest_address);
        auto it1 = cached_queries.find(cont_addr);
        if (it1 != cached_queries.end()) {
            auto it2 = it1->second.find(base);
            if (it2 != it1->second.end()) {
                if (it2->second.raw == loc.raw) {
                    it1->second.erase(it2);
                }
            }
        }
        streamer->Free(query_id);
    }
    impl->pending_unregister.clear();
}

template <typename Traits>
void QueryCacheBase<Traits>::Unregister(QueryCacheBase<Traits>::QueryLocation loc) {
    const auto gen_caching_indexing = [](VAddr cur_addr) {
        return std::make_pair<u64, u32>(cur_addr >> Core::DEVICE_PAGEBITS,
                                        static_cast<u32>(cur_addr & Core::DEVICE_PAGEMASK));
    };
    const auto [streamer_id, query_id] = loc.unpack();
    auto* streamer = impl->streamers[streamer_id];
    if (!streamer) [[unlikely]] {
        return;
    }
    auto* query = streamer->GetQuery(query_id);
    if (!query) [[unlikely]] {
        return;
    }
    const auto [cont_addr, base] = gen_caching_indexing(query->guest_address);
    {
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.find(cont_addr);
        if (it1 != cached_queries.end()) {
            auto it2 = it1->second.find(base);
            if (it2 != it1->second.end() && it2->second.raw == loc.raw) {
                it1->second.erase(it2);
            }
        }
    }
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifyWFI() {
    bool should_sync = false;
    impl->ForEachStreamer(
        [&should_sync](StreamerInterface* streamer) { should_sync |= streamer->HasPendingSync(); });
    if (!should_sync) {
        return;
    }

    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->PresyncWrites(); });
    impl->runtime.Barriers(true);
    impl->ForEachStreamer([](StreamerInterface* streamer) { streamer->SyncWrites(); });
    impl->runtime.Barriers(false);
}

template <typename Traits>
void QueryCacheBase<Traits>::NotifySegment(bool resume) {
    if (resume) {
        impl->runtime.ResumeHostConditionalRendering();
    } else {
        CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
        CounterClose(VideoCommon::QueryType::StreamingByteCount);
        impl->runtime.PauseHostConditionalRendering();
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::AccelerateHostConditionalRendering() {
    NotifyWFI();
    bool qc_dirty = false;
    const auto gen_lookup = [this, &qc_dirty](GPUVAddr address) -> VideoCommon::LookupData {
        auto cpu_addr_opt = gpu_memory->GpuToCpuAddress(address);
        if (!cpu_addr_opt) [[unlikely]] {
            if (UltrahandTraceEnabled()) {
                LOG_WARNING(HW_GPU, "UHTRACE hcr_lookup gpu=0x{:016X} cpu=<unmapped> found=0",
                         address);
            }
            return VideoCommon::LookupData{
                .address = 0,
                .found_query = nullptr,
            };
        }
        VAddr cpu_addr = *cpu_addr_opt;
        const bool pass_trace_lookup =
            UltrahandPassTraceEnabled() && IsUltrahandConditionCpuRange(cpu_addr, 24);
        impl->TrackConditionPage(cpu_addr);
        std::scoped_lock lock(cache_mutex);
        auto it1 = cached_queries.find(cpu_addr >> Core::DEVICE_PAGEBITS);
        if (it1 == cached_queries.end()) {
            if (UltrahandTraceEnabled() || pass_trace_lookup) {
                LOG_WARNING(HW_GPU,
                 "UHTRACE hcr_lookup gpu=0x{:016X} cpu=0x{:016X} found=0 reason=no_page",
                         address, cpu_addr);
            }
            return VideoCommon::LookupData{
                .address = cpu_addr,
                .found_query = nullptr,
            };
        }
        auto& sub_container = it1->second;
        const u32 page_offset = static_cast<u32>(cpu_addr & Core::DEVICE_PAGEMASK);
        auto it_current = sub_container.find(page_offset);
        if (it_current == sub_container.end()) {
            it_current = sub_container.find(page_offset + sizeof(u32));
            if (it_current == sub_container.end()) {
                if (UltrahandTraceEnabled() || pass_trace_lookup) {
                    LOG_WARNING(HW_GPU,
                 "UHTRACE hcr_lookup gpu=0x{:016X} cpu=0x{:016X} found=0 "
                             "reason=no_slot",
                             address, cpu_addr);
                }
                return VideoCommon::LookupData{
                    .address = cpu_addr,
                    .found_query = nullptr,
                };
            }
        }
        auto* query = impl->ObtainQuery(it_current->second);
        if (!query || True(query->flags & QueryFlagBits::IsInvalidated)) {
            if (UltrahandTraceEnabled() || pass_trace_lookup) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE hcr_lookup gpu=0x{:016X} cpu=0x{:016X} found=0 "
                            "reason=invalidated",
                            address, cpu_addr);
            }
            return VideoCommon::LookupData{
                .address = cpu_addr,
                .found_query = nullptr,
            };
        }
        const size_t query_size = True(query->flags & QueryFlagBits::HasTimestamp)
                                      ? sizeof(query->value)
                                      : sizeof(u32);
        u64 current{};
        if (IsStaleGuestQueryWrite(*query,
                                   impl->device_memory.template GetPointer<u8>(
                                       query->guest_address),
                                   query_size, &current)) {
            if (UltrahandTraceEnabled() || pass_trace_lookup) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE hcr_lookup gpu=0x{:016X} cpu=0x{:016X} found=0 "
                            "reason=stale_snapshot flags=0x{:X} value=0x{:016X} "
                            "snapshot=0x{:016X} current=0x{:016X}",
                            address, cpu_addr, static_cast<u32>(query->flags), query->value,
                            query->guest_snapshot, current);
            }
            query->flags |= QueryFlagBits::IsInvalidated;
            sub_container.erase(it_current);
            return VideoCommon::LookupData{
                .address = cpu_addr,
                .found_query = nullptr,
            };
        }
        qc_dirty |= True(query->flags & QueryFlagBits::IsHostManaged) &&
                    False(query->flags & QueryFlagBits::IsGuestSynced);
        if (UltrahandTraceEnabled() || pass_trace_lookup) {
            LOG_WARNING(HW_GPU,
                 "UHTRACE hcr_lookup gpu=0x{:016X} cpu=0x{:016X} found=1 flags=0x{:X} "
                     "host_managed={} guest_synced={} host_synced={} final_synced={} value=0x{:016X}",
                     address, cpu_addr, static_cast<u32>(query->flags),
                     True(query->flags & QueryFlagBits::IsHostManaged),
                     True(query->flags & QueryFlagBits::IsGuestSynced),
                     True(query->flags & QueryFlagBits::IsHostSynced),
                     True(query->flags & QueryFlagBits::IsFinalValueSynced), query->value);
        }
        return VideoCommon::LookupData{
            .address = cpu_addr,
            .found_query = query,
        };
    };

    auto& regs = maxwell3d->regs;
    if (regs.render_enable_override != Tegra::Engines::Maxwell3D::Regs::RenderEnable::Override::UseRenderEnable) {
        impl->runtime.EndHostConditionalRendering();
        return false;
    }
    const ComparisonMode mode = static_cast<ComparisonMode>(regs.render_enable.mode);
    const GPUVAddr address = regs.render_enable.Address();
    const auto tag_render_enable_operand = [](VideoCommon::LookupData& object) {
        if (object.found_query == nullptr ||
            !IsUltrahandConditionCpuRange(object.address, sizeof(object.found_query->value))) {
            return;
        }
        object.found_query->flags |= QueryFlagBits::IsRenderEnableReport;
        if (UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) {
            LOG_WARNING(HW_GPU,
                        "UHTRACE hcr_tag_render_enable cpu=0x{:016X} flags=0x{:X} "
                        "value=0x{:016X}",
                        object.address, static_cast<u32>(object.found_query->flags),
                        object.found_query->value);
        }
    };
    const auto address_cpu_opt = gpu_memory->GpuToCpuAddress(address);
    const bool pass_trace_hcr =
        address_cpu_opt && UltrahandPassTraceEnabled() &&
        IsUltrahandConditionCpuRange(*address_cpu_opt, 24);
    if (UltrahandTraceEnabled() || pass_trace_hcr) {
        LOG_WARNING(HW_GPU, "UHTRACE hcr_start gpu=0x{:016X} override={} mode={}", address,
                 static_cast<u32>(regs.render_enable_override), static_cast<u32>(mode));
    }
    switch (mode) {
    case ComparisonMode::True:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::False:
        impl->runtime.EndHostConditionalRendering();
        return false;
    case ComparisonMode::Conditional: {
        VideoCommon::LookupData object_1{gen_lookup(address)};
        const bool result = impl->runtime.HostConditionalRenderingCompareValue(object_1, qc_dirty);
        if (UltrahandTraceEnabled() || pass_trace_hcr) {
            LOG_WARNING(HW_GPU, "UHTRACE hcr_result mode=conditional result={} qc_dirty={}", result,
                     qc_dirty);
        }
        return result;
    }
    case ComparisonMode::IfEqual: {
        impl->render_enable_compare_addresses.insert(address);
        VideoCommon::LookupData object_1{gen_lookup(address)};
        tag_render_enable_operand(object_1);
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        const bool result =
            impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2, qc_dirty, true);
        if (UltrahandTraceEnabled() || pass_trace_hcr) {
            LOG_WARNING(HW_GPU, "UHTRACE hcr_result mode=equal result={} qc_dirty={}", result,
                     qc_dirty);
        }
        return result;
    }
    case ComparisonMode::IfNotEqual: {
        impl->render_enable_compare_addresses.insert(address);
        VideoCommon::LookupData object_1{gen_lookup(address)};
        tag_render_enable_operand(object_1);
        VideoCommon::LookupData object_2{gen_lookup(address + 16)};
        const bool result = impl->runtime.HostConditionalRenderingCompareValues(object_1, object_2,
                                                                               qc_dirty, false);
        if (UltrahandTraceEnabled() || pass_trace_hcr) {
            LOG_WARNING(HW_GPU, "UHTRACE hcr_result mode=not_equal result={} qc_dirty={}", result,
                     qc_dirty);
        }
        return result;
    }
    default:
        return false;
    }
}

// Async downloads
template <typename Traits>
void QueryCacheBase<Traits>::CommitAsyncFlushes() {
    // Make sure to have the results synced in Host.
    NotifyWFI();

    u64 mask{};
    {
        std::scoped_lock lk(impl->flush_guard);
        impl->ForEachStreamer([&mask](StreamerInterface* streamer) {
            bool local_result = streamer->HasUnsyncedQueries();
            if (local_result) {
                mask |= 1ULL << streamer->GetId();
            }
        });
        impl->flushes_pending.push_back(mask);
    }
    std::function<void()> func([this] { UnregisterPending(); });
    impl->rasterizer.SyncOperation(std::move(func));
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependentMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PushUnsyncedQueries();
        });
    }
}

template <typename Traits>
bool QueryCacheBase<Traits>::HasUncommittedFlushes() const {
    bool result = false;
    impl->ForEachStreamer([&result](StreamerInterface* streamer) {
        result |= streamer->HasUnsyncedQueries();
        return result;
    });
    return result;
}

template <typename Traits>
bool QueryCacheBase<Traits>::ShouldWaitAsyncFlushes() {
    std::scoped_lock lk(impl->flush_guard);
    return !impl->flushes_pending.empty() && impl->flushes_pending.front() != 0ULL;
}

template <typename Traits>
void QueryCacheBase<Traits>::PopAsyncFlushes() {
    u64 mask;
    {
        std::scoped_lock lk(impl->flush_guard);
        mask = impl->flushes_pending.front();
        impl->flushes_pending.pop_front();
    }
    if (mask == 0) {
        return;
    }
    u64 ran_mask = ~mask;
    while (mask) {
        impl->ForEachStreamerIn(mask, [&mask, &ran_mask](StreamerInterface* streamer) {
            u64 dep_mask = streamer->GetDependenceMask();
            if ((dep_mask & ~ran_mask) != 0) {
                return;
            }
            u64 index = streamer->GetId();
            ran_mask |= (1ULL << index);
            mask &= ~(1ULL << index);
            streamer->PopUnsyncedQueries();
        });
    }
}

// Invalidation

template <typename Traits>
void QueryCacheBase<Traits>::InvalidateQuery(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return;
    }
    if (UltrahandTraceEnabled()) {
        LOG_WARNING(HW_GPU,
                    "UHTRACE qc_invalidate cpu=0x{:016X} flags=0x{:X} value=0x{:016X}",
                    query_base->guest_address, static_cast<u32>(query_base->flags),
                    query_base->value);
    }
    query_base->flags |= QueryFlagBits::IsInvalidated;
}

template <typename Traits>
bool QueryCacheBase<Traits>::IsQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    if (True(query_base->flags & QueryFlagBits::IsInvalidated) ||
        True(query_base->flags & QueryFlagBits::IsRewritten)) {
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
bool QueryCacheBase<Traits>::SemiFlushQueryDirty(QueryCacheBase<Traits>::QueryLocation location) {
    auto* query_base = impl->ObtainQuery(location);
    if (!query_base) {
        return false;
    }
    if (True(query_base->flags & QueryFlagBits::IsFinalValueSynced) &&
        False(query_base->flags & QueryFlagBits::IsGuestSynced)) {
        auto* ptr = impl->device_memory.template GetPointer<u8>(query_base->guest_address);
        if (True(query_base->flags & QueryFlagBits::HasTimestamp)) {
            u64 current{};
            if (IsStaleGuestQueryWrite(*query_base, ptr, sizeof(query_base->value), &current)) {
                if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                    IsUltrahandConditionCpuRange(query_base->guest_address,
                                                 sizeof(query_base->value))) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE qc_semiflush_guest_skip_stale cpu=0x{:016X} size=8 "
                                "flags=0x{:X} value=0x{:016X} snapshot=0x{:016X} "
                                "current=0x{:016X}",
                                query_base->guest_address, static_cast<u32>(query_base->flags),
                                query_base->value, query_base->guest_snapshot, current);
                }
                query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
                Unregister(location);
                return false;
            }
            if (IsUnsafeRenderEnableGuestWrite(*query_base, query_base->value, current)) {
                if (UltrahandTraceEnabled()) {
                    LOG_WARNING(HW_GPU,
                                "UHTRACE qc_semiflush_guest_skip_render_enable cpu=0x{:016X} "
                                "size=8 flags=0x{:X} value=0x{:016X} snapshot=0x{:016X} "
                                "current=0x{:016X}",
                                query_base->guest_address, static_cast<u32>(query_base->flags),
                                query_base->value, query_base->guest_snapshot, current);
                }
                query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
                Unregister(location);
                return false;
            }
            if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(query_base->value))) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE qc_semiflush_guest_write cpu=0x{:016X} size=8 "
                            "flags=0x{:X} value=0x{:016X} snapshot=0x{:016X} "
                            "current=0x{:016X} timestamp=1",
                            query_base->guest_address, static_cast<u32>(query_base->flags),
                            query_base->value, query_base->guest_snapshot, current);
            }
            std::memcpy(ptr, &query_base->value, sizeof(query_base->value));
            query_base->flags |= QueryFlagBits::IsGuestSynced;
            Unregister(location);
            return false;
        }
        u32 value_l = static_cast<u32>(query_base->value);
        u64 current{};
        if (IsStaleGuestQueryWrite(*query_base, ptr, sizeof(value_l), &current)) {
            if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
                IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(value_l))) {
                LOG_WARNING(HW_GPU,
                            "UHTRACE qc_semiflush_guest_skip_stale cpu=0x{:016X} size=4 "
                            "flags=0x{:X} value=0x{:08X} snapshot=0x{:016X} current=0x{:016X}",
                            query_base->guest_address, static_cast<u32>(query_base->flags),
                            value_l, query_base->guest_snapshot, current);
            }
            query_base->flags |= QueryFlagBits::IsGuestSynced | QueryFlagBits::IsInvalidated;
            Unregister(location);
            return false;
        }
        if ((UltrahandTraceEnabled() || UltrahandPassTraceEnabled()) &&
            IsUltrahandConditionCpuRange(query_base->guest_address, sizeof(value_l))) {
            LOG_WARNING(HW_GPU,
                        "UHTRACE qc_semiflush_guest_write cpu=0x{:016X} size=4 "
                        "flags=0x{:X} value=0x{:08X} timestamp=0",
                        query_base->guest_address, static_cast<u32>(query_base->flags), value_l);
        }
        std::memcpy(ptr, &value_l, sizeof(value_l));
        query_base->flags |= QueryFlagBits::IsGuestSynced;
        Unregister(location);
        return false;
    }
    return True(query_base->flags & QueryFlagBits::IsHostManaged) &&
           False(query_base->flags & QueryFlagBits::IsGuestSynced);
}

template <typename Traits>
void QueryCacheBase<Traits>::RequestGuestHostSync() {
    impl->rasterizer.ReleaseFences();
}

} // namespace VideoCommon
