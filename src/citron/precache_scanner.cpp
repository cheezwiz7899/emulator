// SPDX-FileCopyrightText: 2025 citron Emulator Project
// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GMainWindow::OnGameListPreCacheShaders — the boot-time shader pre-cache scanner.
// Extracted from main.cpp, where it grew to ~1,400 lines across several rounds of
// work: the original BNSH/GRSC/SPH scanner, real previous-stage-stores
// guess refinement, and a broad round of additional container-format support —
// Yaz0, SARC, pairtable, ARC, XC2's arh/ard+xbc1, CPK+CRILAYLA, and MPR's
// RFRM/MTRL material archives. See docs/precache-scanner/FINDINGS.md and
// docs/precache-scanner/HANDOFF.md for the full investigation behind the format
// support, and handoff_10/11/12 (shader-precache investigation) for the
// previous-stage-stores work. Kept as a single translation unit rather than
// split further — the internal lambdas (process_blob, try_translate_at,
// process_stage_offset, process_bnsh_at) are tightly coupled via [&] capture of
// this function's own locals, and untangling that into independent, separately
// testable pieces is a real future improvement but a materially bigger and
// riskier undertaking than the file-level extraction done here.
//
// FileWorkItem, ScanUnit, and PreviousStageStoresSnapshot are hoisted to file
// scope (anonymous namespace) below rather than left as locals nested hundreds
// of lines into the function, unlike the lambdas above — plain data structs
// with no captures, so hoisting them is behavior-preserving, and each is
// substantial and independently meaningful enough to read on its own.

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <system_error>
#include <unordered_set>

#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <fmt/format.h>

#include "citron/main.h"

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/thread_worker.h"
#include "common/crilayla_compression.h"
#include "common/xbc1_compression.h"
#include "common/yaz0_compression.h"
#include "common/zlib_compression.h"
#include "common/zstd_compression.h"

#include "core/loader/loader.h"
#include "core/loader/nca.h"
#include "core/core.h"
#include "core/file_sys/arc_archive.h"
#include "core/file_sys/card_image.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/cpk_archive.h"
#include "core/file_sys/mpr_material_archive.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/sarc_archive.h"
#include "core/file_sys/submission_package.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/file_sys/xc2_arh_archive.h"

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/decode.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/program_header.h"
#include "shader_recompiler/runtime_info.h"

#include "video_core/precache_cfg_artifact.h"
#include "video_core/precache_frontend_artifact.h"
#include "video_core/speculative_shader_environment.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"
#include "video_core/precache_compiler_target.h"
#include "video_core/shader_program_identity.h"
#include "video_core/spirv_cache.h"

namespace {

// The scanner's basic unit of work: one real sub-file to scan, whichever
// container (if any) it came from. arc_range/arh_range/cpk_range/mpr_range are
// mutually exclusive — nullopt means "the whole RomFS file, unwrapped", set
// means a ranged read into a larger archive/index-described file. Every
// container this scanner supports (ARC, the .arh/.ard pair, CPK, MPR) expands
// into a list of these instead of ever queuing a multi-gigabyte file as one
// work item — real parallelism, and no full-file loads for anything large.
struct FileWorkItem {
    FileSys::VirtualFile file;
    std::optional<FileSys::ArcSubFile> arc_range; // nullopt = whole file, as before.
    std::optional<FileSys::ArhSubFile> arh_range; // set only for .arh/.ard-paired entries below.
    std::optional<FileSys::CpkFileEntry> cpk_range; // set only for CPK archive entries below.
    std::optional<FileSys::MprShaderSource> mpr_range; // set only for MPR MaterialArchive entries below.
};

// Scanner runtime-state snapshot. Mirrors the fields MakeRuntimeInfo()
// (vk_pipeline_cache.cpp) pulls from a real previous_program when deriving a real
// pipeline's previous_stage_stores -- see that function for the reference
// derivation this mirrors field-for-field. Deliberately NOT holding onto the
// translated Shader::IR::Program itself: its IR::Block/IR::Inst graph lives in
// per-attempt ObjectPools (see try_translate_at, within the function below) that
// go out of scope as soon as that lambda returns, so keeping a Program around
// across sibling-stage calls would dangle. VaryingState (a bitset<512> wrapper)
// and std::map<IR::Attribute,IR::Attribute> are both plain value types with no
// pool dependency, so copying just these three fields out avoids that trap
// entirely instead of trying to extend any pool's lifetime.
struct PreviousStageStoresSnapshot {
    Shader::VaryingState stores{};
    std::map<Shader::IR::Attribute, Shader::IR::Attribute> legacy_stores_mapping{};
    Shader::VaryingState passthrough{};
    bool is_geometry_passthrough{};
};

// A successful speculative translation is reusable only when its complete
// pipeline context matches. Keeping the resulting successor state alongside the
// dedup key is essential: callers chain this state into the next BNSH stage.
struct PrecacheStageResult {
    PreviousStageStoresSnapshot stage_snapshot{};
    Shader::Backend::Bindings end_binding{};
};

// One already-unwrapped region of bytes ready for the BNSH/GRSC/SPH scan below —
// either a whole post-decompression file, one named entry out of a SARC/U8
// archive, or one block out of a pairtable-wrapped file. Built as a list rather
// than recursing so the existing scan/classify logic runs unmodified, once per
// unit -- a plain file still produces exactly one unit, so this is a no-op for
// every title that doesn't need container unwrapping at this stage.
struct ScanUnit {
    std::vector<u8> owned; // Empty if data points at decompressed_storage/raw instead.
    const u8* data;
    size_t sz;
    std::string entry_name; // Empty for a plain (non-SARC-entry) unit.
    bool scan_full_for_raw_sph = false; // Set only for pairtable-derived blocks --
                                         // see the pairtable-detection block in
                                         // the function below for why those
                                         // specifically need a full-offset scan
                                         // rather than an offset-0 check.
};

} // namespace


// ── GPL: Pre-cache Shaders handler ────────────────────────────────────────
void GMainWindow::OnGameListPreCacheShaders(u64 program_id,
                                             const std::string& game_path) {
    if (program_id == 0 || game_path.empty()) return;

    if (precache_target_capture) {
        QMessageBox::information(this, tr("Pre-cache Shaders"),
                                 tr("A compiler-target capture is already running."));
        return;
    }
    if (emulation_running) {
        QMessageBox::warning(this, tr("Pre-cache Shaders"),
                             tr("Stop emulation before starting shader pre-cache."));
        return;
    }

    const auto shader_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::ShaderDir);
    const auto cache_dir = shader_dir / fmt::format("{:016x}", program_id);
    if (!Common::FS::CreateDirs(cache_dir)) return;
    BeginPrecacheTargetCapture(program_id, game_path,
                               cache_dir / "precache_compiler_target.bin");
}

void GMainWindow::BeginPrecacheTargetCapture(u64 program_id, const std::string& game_path,
                                             const std::filesystem::path& target_path) {
    // A previous capture may belong to a different driver or renderer configuration.
    // Remove it so the poll below can only accept the target written by this boot.
    std::error_code error;
    std::filesystem::remove(target_path, error);
    if (error) {
        LOG_ERROR(Frontend, "Pre-cache could not clear stale compiler target {}: {}",
                  Common::FS::PathToUTF8String(target_path), error.message());
        QMessageBox::warning(this, tr("Pre-cache Shaders"),
                             tr("Could not refresh the Vulkan compiler target."));
        return;
    }

    precache_target_capture = PrecacheTargetCaptureRequest{
        .program_id = program_id,
        .game_path = game_path,
        .target_path = target_path,
    };
    precache_target_boot_ready = false;
    precache_target_capture_dialog = new QProgressDialog(
        tr("Starting the title briefly to capture the Vulkan compiler target..."), tr("Cancel"),
        0, 0, this);
    precache_target_capture_dialog->setWindowTitle(tr("Pre-cache Shaders"));
    precache_target_capture_dialog->setWindowModality(Qt::WindowModal);
    precache_target_capture_dialog->setAutoClose(false);
    precache_target_capture_dialog->show();

    LOG_INFO(Frontend, "Pre-cache starting automatic Vulkan compiler-target capture for {:016x}",
             program_id);

    // BootGameFromList performs substantial renderer setup synchronously, and may
    // process Qt events while doing so. Starting the poll timer before it returns
    // lets PollPrecacheTargetCapture run re-entrantly in the middle of LoadROM:
    // the target has already been written, but emulation_running is not set yet.
    // That used to start the multi-minute scanner on the unfinished boot stack,
    // then let the damaged boot resume afterward. Arm all asynchronous handling
    // only after the synchronous portion of boot has completed.
    BootGameFromList(QString::fromStdString(game_path), StartGameType::Normal);
    if (precache_target_capture && !emulation_running) {
        LOG_ERROR(Frontend, "Pre-cache target-capture boot did not start for {:016x}", program_id);
        FinishPrecacheTargetCapture(false);
    } else if (precache_target_capture) {
        connect(precache_target_capture_dialog, &QProgressDialog::canceled, this, [this] {
            if (!precache_target_capture) return;
            LOG_INFO(Frontend, "Pre-cache compiler-target capture cancelled by user");
            if (emulation_running) ShutdownGame();
            FinishPrecacheTargetCapture(false);
        });

        if (precache_target_capture_dialog->wasCanceled()) {
            LOG_INFO(Frontend, "Pre-cache compiler-target capture cancelled during boot");
            ShutdownGame();
            FinishPrecacheTargetCapture(false);
            return;
        }

        disconnect(&precache_target_capture_timer, nullptr, this, nullptr);
        connect(&precache_target_capture_timer, &QTimer::timeout, this,
                &GMainWindow::PollPrecacheTargetCapture);
        PollPrecacheTargetCapture();
        if (precache_target_capture) {
            precache_target_capture_timer.start(200);
        }
    }
}

void GMainWindow::PollPrecacheTargetCapture() {
    if (!precache_target_capture) return;

    const auto request = *precache_target_capture;
    if (precache_target_boot_ready &&
        VideoCommon::LoadPrecacheCompilerTarget(request.target_path)) {
        // Static ROM scanning cannot know real cbuf, interface, or scheduling
        // state. Keep the title alive briefly to collect exact launch requests;
        // these are safe entries for the following real play session.
        constexpr int kWarmupPolls = 75; // 15 seconds at the 200 ms poll interval.
        if (++precache_target_capture->ready_poll_count < kWarmupPolls) return;
        LOG_INFO(Frontend,
                 "Pre-cache captured Vulkan compiler target and 15-second exact warmup for {:016x}; "
                 "stopping boot",
                 request.program_id);
        precache_target_capture_timer.stop();
        // Do this while the renderer still exists. The warmup's real shaders
        // include cbuf/interface/resource state static ROM scanning cannot
        // know, and are valid exact entries for every title. Explicit flush
        // also prevents a queued save from racing the scanner's Load/Save.
        if (emulation_running) {
            system->GPU().Renderer().FlushShaderCaches();
        }
        if (emulation_running) ShutdownGame();
        FinishPrecacheTargetCapture(true);
        return;
    }

    if (!emulation_running) {
        LOG_ERROR(Frontend, "Pre-cache target-capture boot stopped before target creation for {:016x}",
                  request.program_id);
        FinishPrecacheTargetCapture(false);
        return;
    }

    // File creation happens during early Vulkan setup. Wait instead for the first
    // displayed frame, where GPU and guest-memory startup are both complete.
    if (++precache_target_capture->poll_count >= 900) {
        LOG_ERROR(Frontend, "Pre-cache timed out waiting for Vulkan compiler target for {:016x}",
                  request.program_id);
        if (emulation_running) ShutdownGame();
        FinishPrecacheTargetCapture(false);
    }
}

void GMainWindow::FinishPrecacheTargetCapture(bool start_scan) {
    if (!precache_target_capture) return;

    const auto request = std::move(*precache_target_capture);
    precache_target_capture.reset();
    precache_target_boot_ready = false;
    precache_target_capture_timer.stop();
    disconnect(&precache_target_capture_timer, nullptr, this, nullptr);
    if (precache_target_capture_dialog) {
        precache_target_capture_dialog->disconnect(this);
        precache_target_capture_dialog->close();
        precache_target_capture_dialog->deleteLater();
        precache_target_capture_dialog = nullptr;
    }

    if (!start_scan) {
        QMessageBox::warning(this, tr("Pre-cache Shaders"),
                             tr("Citron could not capture a Vulkan compiler target.\n"
                                "Make sure Vulkan is selected and the title can boot."));
        return;
    }

    StartPrecacheScan(request.program_id, request.game_path,
                      std::move(request.resolved_romfs_roots));
}

void GMainWindow::StartPrecacheScan(u64 program_id, const std::string& game_path,
                                    std::vector<FileSys::VirtualDir> resolved_romfs_roots) {
    if (program_id == 0 || game_path.empty()) return;

    const auto shader_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::ShaderDir);
    const auto cache_dir  = shader_dir / fmt::format("{:016x}", program_id);
    if (!Common::FS::CreateDirs(cache_dir)) return;
    const auto spirv_path = cache_dir / "spirv_cache.bin";
    const auto target_path = cache_dir / "precache_compiler_target.bin";
    if (!VideoCommon::LoadPrecacheCompilerTarget(target_path)) {
        LOG_ERROR(Render_Vulkan,
                  "PreCacheShaders: refusing to scan without a valid real Vulkan compiler target");
        QMessageBox::warning(this, tr("Pre-cache Shaders"),
                             tr("The Vulkan compiler target is missing or invalid."));
        return;
    }

    const int kMaxSamples = 8;
    // Keep per-blob tracing bounded. The scan summary contains aggregate
    // coverage and rejection data; thousands of expected alignment failures
    // make the user log too large to inspect and add avoidable worker I/O.
    const int kMaxDiagBlobs = 8;
    // Must match video_core/precache_cfg_artifact.cpp. Keep a pathological
        // ROM scan from turning one all-or-nothing artifact file into a save miss.
    constexpr size_t kMaxPersistedCfgArtifacts = 131072;
    constexpr size_t kMaxPersistedFrontendArtifacts = 131072;

    struct ScanState {
        std::atomic<int>  files_total{0};
        std::atomic<int>  files_processed{0};
        std::atomic<int>  shaders_found{0};
        std::atomic<int>  shaders_translated{0};
        // Exact final modules and successful translations whose CFG required a
        // nonzero candidate scheduler alignment. The latter is extraction
        // telemetry only: a static blob cannot prove that alignment at runtime.
        std::atomic<int>  final_entries{0};
        std::atomic<int>  alignment_translated_entries{0};
        std::atomic<int>  cfg_artifact_candidates{0};
        std::atomic<int>  cfg_artifact_stored{0};
        std::atomic<int>  cfg_artifact_cbuf_rejections{0};
        std::atomic<int>  cfg_artifact_branches_rejections{0};
        std::atomic<int>  cfg_artifact_capacity_rejections{0};
        std::atomic<int>  frontend_artifact_candidates{0};
        std::atomic<int>  frontend_artifact_stored{0};
        std::atomic<int>  frontend_artifact_info_rejections{0};
        std::atomic<int>  frontend_artifact_pseudo_rejections{0};
        std::atomic<int>  frontend_artifact_value_rejections{0};
        std::atomic<int>  frontend_artifact_graph_rejections{0};
        std::atomic<int>  frontend_artifact_capacity_rejections{0};
        // Scanner-returned zero is unknown state, never an observed zero.
        // Report dependency classes separately so next scan can measure which
        // candidates need a real contract/template boundary.
        std::atomic<int>  unknown_cbuf_candidates{0};
        std::atomic<int>  unknown_texture_candidates{0};
        std::atomic<int>  unknown_viewport_candidates{0};
        std::atomic<int>  unknown_compute_candidates{0};
        std::atomic<int>  unknown_hle_macro_candidates{0};
        std::atomic<int>  unknown_interface_candidates{0};
        std::atomic<int>  unknown_scheduling_alignment_candidates{0};
        std::atomic<int>  final_cbuf_contract_rejections{0};
        std::atomic<int>  final_texture_contract_rejections{0};
        std::atomic<int>  final_viewport_contract_rejections{0};
        std::atomic<int>  final_compute_contract_rejections{0};
        std::atomic<int>  final_hle_macro_contract_rejections{0};
        std::atomic<int>  final_interface_contract_rejections{0};
        std::atomic<int>  final_scheduling_alignment_contract_rejections{0};
        std::atomic<int>  final_identity_contract_rejections{0};
        // BuildProgramTemplate() is the proposed scanner artifact boundary.
        // These counters distinguish candidates that already consumed synthetic
        // Environment state before finalization from candidates that could be
        // represented by a future immutable frontend template.
        std::atomic<int>  template_boundary_eligible{0};
        std::atomic<int>  template_boundary_cbuf_rejections{0};
        std::atomic<int>  template_boundary_texture_rejections{0};
        std::atomic<int>  template_boundary_viewport_rejections{0};
        std::atomic<int>  template_boundary_compute_rejections{0};
        std::atomic<int>  template_boundary_hle_macro_rejections{0};
        std::atomic<int>  template_boundary_interface_rejections{0};
        std::atomic<int>  template_boundary_scheduling_alignment_rejections{0};
        // Actual BuildProgramTemplate() query manifest. Unlike the unknown-
        // dependency counters, this records state that was consumed even when
        // the scanner happened to have a concrete fallback value for it.
        std::atomic<int>  frontend_manifest_cbuf{0};
        std::atomic<int>  frontend_manifest_resource{0};
        std::atomic<int>  frontend_manifest_viewport{0};
        std::atomic<int>  frontend_manifest_compute{0};
        std::atomic<int>  frontend_manifest_hle{0};
        std::atomic<int>  frontend_manifest_interface{0};
        std::atomic<int>  shaders_failed{0};
        // Sum of worker CPU-wall time, not scan wall time: workers run in
        // parallel. This identifies the next template boundary without
        // falsely presenting concurrent work as a serial duration.
        std::atomic<u64> cfg_us{0};
        std::atomic<u64> template_us{0};
        std::atomic<u64> finalize_us{0};
        std::atomic<u64> emit_us{0};
        std::atomic<bool> cancelled{false};
        std::string       error_message;
        // Breakdown of what each file's first bytes matched, to see empirically
        // which container format (if any) the ROM's shader files actually use,
        // rather than guessing blind. bnsh_matched/raw_matched count files whose
        // magic/header shape was recognized at all (regardless of whether any
        // shader inside them later passed the Maxwell SPH validation and
        // produced a candidate); unrecognized is everything else.
        std::atomic<int>  bnsh_matched{0};
        // zstd-compressed files (TotK-style shared-dictionary scheme): how many
        // were successfully decompressed (with or without a dictionary) vs.
        // failed (frame claims a Dictionary_ID we don't have loaded, or the
        // frame itself is corrupt/unsupported).
        std::atomic<int>  zstd_decompressed{0};
        std::atomic<int>  zstd_failed{0};
        // Temporary diagnostic: log the parsed SPH fields + outcome for the
        // first several blobs that reach process_blob(), regardless of
        // pass/fail, so a real run gives concrete evidence of what's actually
        // happening to extracted candidates instead of more guessing.
        std::atomic<int>  diag_blobs_logged{0};
        std::atomic<int>  raw_matched{0};
        std::atomic<int>  unrecognized{0};
        // A handful of unrecognized files' first bytes, logged once so their
        // actual format can be identified instead of guessed. Capped so a
        // huge ROM doesn't spam the log.
        std::mutex        sample_mutex;
        int               samples_logged{0};
        std::mutex        cfg_artifact_mutex;
        std::vector<VideoCommon::PrecacheCfgArtifact> cfg_artifacts;
        std::unordered_set<VideoCommon::PrecacheCfgArtifactKey,
                           VideoCommon::PrecacheCfgArtifactKeyHash>
            cfg_artifact_keys;
        std::array<int, 7> cfg_artifact_stages{};
        std::mutex        frontend_artifact_mutex;
        std::vector<VideoCommon::PrecacheFrontendArtifactRecord> frontend_artifacts;
        std::unordered_set<VideoCommon::PrecacheFrontendArtifactKey,
                           VideoCommon::PrecacheFrontendArtifactKeyHash>
            frontend_artifact_keys;
        std::array<int, 7> frontend_artifact_stages{};
    };
    auto state = std::make_shared<ScanState>();

    struct FinalResult { int translated, failed; std::string error; bool cancelled; };
    FinalResult final_result{};

    QProgressDialog progress(tr("Preparing resolved RomFS scan..."),
                             tr("Cancel"), 0, 0, this);
    progress.setWindowTitle(tr("Pre-cache Shaders"));
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setAutoClose(false);
    progress.setAutoReset(false);

    auto local_vfs = std::make_shared<FileSys::RealVfsFilesystem>();
    const auto game_file = local_vfs->OpenFile(game_path, FileSys::OpenMode::Read);

    auto worker = [state, game_path, spirv_path, cache_dir, target_path, game_file,
                   resolved_romfs_roots = std::move(resolved_romfs_roots)]() mutable {
        // Prefer the resolved view captured while the short boot was alive. It
        // contains the same base/update/LayeredFS content the title actually
        // ran, plus enabled DLC roots. The fallback keeps direct-file scanning
        // available if a title fails before its first frame.
        std::vector<FileSys::VirtualDir> romfs_roots = std::move(resolved_romfs_roots);
        FileSys::VirtualFile romfs_raw;
        const auto try_nca = [&]() {
            FileSys::NCA nca{game_file};
            if (nca.GetStatus() != Loader::ResultStatus::Success) return;
            if (nca.GetType()   != FileSys::NCAContentType::Program) return;
            romfs_raw = nca.GetRomFS();
        };
        const auto try_nsp = [&]() {
            FileSys::NSP nsp{game_file};
            if (nsp.GetStatus() != Loader::ResultStatus::Success) return;
            for (const auto& nca : nsp.GetNCAsCollapsed()) {
                if (!nca || nca->GetStatus() != Loader::ResultStatus::Success) continue;
                if (nca->GetType() != FileSys::NCAContentType::Program) continue;
                romfs_raw = nca->GetRomFS();
                if (romfs_raw) return;
            }
        };
        const auto try_xci = [&]() {
            FileSys::XCI xci{game_file};
            if (xci.GetStatus() != Loader::ResultStatus::Success) return;
            const auto nca = xci.GetNCAByType(FileSys::NCAContentType::Program);
            if (!nca || nca->GetStatus() != Loader::ResultStatus::Success) return;
            romfs_raw = nca->GetRomFS();
        };

        const std::string ext = [&]{
            const auto dot = game_path.rfind('.');
            if (dot == std::string::npos) return std::string{};
            auto e = game_path.substr(dot+1);
            for (auto& ch : e) ch = static_cast<char>(std::tolower(ch));
            return e;
        }();
        if (romfs_roots.empty()) {
            if      (ext=="nca")               try_nca();
            else if (ext=="nsp"||ext=="nsz")   try_nsp();
            else if (ext=="xci"||ext=="xcz")   try_xci();
            else { try_nca(); if (!romfs_raw) try_nsp(); if (!romfs_raw) try_xci(); }

            if (!romfs_raw) {
                state->error_message =
                    "Could not mount RomFS. Ensure prod.keys is installed.";
                LOG_ERROR(Render_Vulkan, "PreCacheShaders: {}", state->error_message);
                return;
            }
            if (const auto romfs = FileSys::ExtractRomFS(romfs_raw)) {
                romfs_roots.push_back(romfs);
            } else {
                state->error_message = "Failed to extract RomFS.";
                LOG_ERROR(Render_Vulkan, "PreCacheShaders: {}", state->error_message);
                return;
            }
        } else {
            LOG_INFO(Render_Vulkan,
                     "PreCacheShaders: scanning {} boot-resolved RomFS root(s); no remount needed",
                     romfs_roots.size());
        }

        // Some titles (e.g. Tears of the Kingdom) compress most RomFS assets
        // against a shared dictionary rather than standalone. The dictionaries
        // themselves live inside the SAME title's RomFS — grabbed fresh here
        // per scan, not hardcoded — as a SARC archive at Pack/ZsDic.pack.zs,
        // itself zstd-compressed but WITHOUT a dictionary (it's the bootstrap).
        // Titles that don't use this scheme simply won't have this file; the
        // map stays empty and any dictionary-compressed file found later is
        // just counted as a decompression failure rather than crashing.
        std::unordered_map<u32, std::vector<u8>> dictionaries_by_id;
        // Keeps the decompressed ZsDic.pack SARC's owned buffer alive for the
        // rest of the scan — dictionaries_by_id copies out of it once below,
        // so this only needs to survive that copy, but keeping it named makes
        // the lifetime obvious rather than relying on a temporary.
        std::optional<FileSys::SarcArchive> zsdic_sarc;
        FileSys::VirtualFile zsdic_file;
        for (const auto& root : romfs_roots) {
            if (root && (zsdic_file = root->GetFileRelative("Pack/ZsDic.pack.zs"))) {
                break;
            }
        }
        if (zsdic_file) {
            const auto zsdic_compressed = zsdic_file->ReadAllBytes();
            auto zsdic_decompressed = Common::Compression::DecompressDataZSTD(zsdic_compressed);
            if (zsdic_decompressed.empty()) {
                LOG_ERROR(Render_Vulkan,
                          "PreCacheShaders: found Pack/ZsDic.pack.zs but failed to "
                          "decompress it (expected no dictionary — this file bootstraps "
                          "every other dictionary, so it can't need one itself)");
            } else {
                zsdic_sarc = FileSys::SarcArchive::Parse(std::move(zsdic_decompressed));
                if (!zsdic_sarc->Ok()) {
                    LOG_ERROR(Render_Vulkan,
                              "PreCacheShaders: Pack/ZsDic.pack.zs decompressed but isn't a "
                              "valid SARC archive — dictionary format may have changed");
                } else {
                    for (const auto& entry : zsdic_sarc->Entries()) {
                        const u32 dict_id = Common::Compression::GetZSTDDictionaryID(entry.data);
                        if (dict_id == 0) continue; // Not a recognizable raw-content dictionary.
                        dictionaries_by_id.emplace(
                            dict_id, std::vector<u8>{entry.data.begin(), entry.data.end()});
                        LOG_INFO(Render_Vulkan,
                                 "PreCacheShaders: loaded dictionary '{}' (id={}, {} bytes)",
                                 entry.name, dict_id, entry.data.size());
                    }
                }
            }
        }

        std::vector<FileSys::VirtualFile> files;
        std::function<void(const FileSys::VirtualDir&)> walk =
            [&](const FileSys::VirtualDir& dir) {
                if (!dir) return;
                for (const auto& f : dir->GetFiles())         files.push_back(f);
                for (const auto& sub : dir->GetSubdirectories()) walk(sub);
            };
        for (const auto& root : romfs_roots) {
            walk(root);
        }

        // Temporary debug aid: set CITRON_PRECACHE_FILTER to a substring (e.g.
        // "grass" or a specific file name) before launching citron to restrict
        // the scan to only files whose full RomFS path contains it. Lets a
        // known-good sample be iterated on in seconds instead of waiting
        // several minutes for a full ~300k-file ROM scan on every rebuild.
        // Unset (the default) scans everything, as before.
        if (const char* filter = std::getenv("CITRON_PRECACHE_FILTER"); filter && *filter) {
            const std::string needle(filter);
            std::vector<FileSys::VirtualFile> filtered;
            filtered.reserve(files.size());
            for (auto& f : files) {
                if (f->GetFullPath().find(needle) != std::string::npos) {
                    filtered.push_back(f);
                }
            }
            LOG_INFO(Render_Vulkan,
                     "PreCacheShaders: CITRON_PRECACHE_FILTER='{}' active — {} of {} files matched",
                     needle, filtered.size(), files.size());
            files = std::move(filtered);
        }

        // Some titles pack virtually their entire game into one monolithic
        // RomFS file rather than the thousands of individually-sized real
        // files every other title has — Smash Ultimate's data.arc (10+ GB)
        // is the one this was written for. Treating that as a single work
        // item means one worker thread reads the whole thing into memory
        // and scans it serially while the rest of the pool sits idle; see
        // docs/precache-scanner/FINDINGS.md for the investigation this
        // fixes. TryEnumerateArcSubFiles detects the format by its own
        // magic (not by filename) and returns the real, already
        // individually-compressed sub-file boundaries packed inside, each
        // of which becomes its own properly parallelized work item below
        // instead. Every other title's files aren't a recognized ARC, so
        // TryEnumerateArcSubFiles returns empty and this is a no-op.
        std::vector<FileWorkItem> work_items;
        work_items.reserve(files.size());

        // Xenoblade Chronicles 2 (and plausibly sibling Monolith Soft
        // titles) splits its real data across a small ".arh" index file and
        // a separate, large, paired ".ard" data file with the same base
        // name — unlike every other container this scanner understands,
        // the index and the data it describes are two different RomFS
        // files, so this can't be handled by TryEnumerateArcSubFiles-style
        // self-contained detection. Built as a name-based lookup (find each
        // ".arh" file's ".ard" sibling by path) rather than folding into
        // the walk above, since pairing needs the full file list to search
        // rather than a single file's own bytes. See
        // docs/precache-scanner/FINDINGS.md section 7.
        std::vector<bool> consumed_by_arh(files.size(), false);
        for (size_t i = 0; i < files.size(); ++i) {
            const auto& arh_file = files[i];
            const std::string full_path = arh_file->GetFullPath();
            if (full_path.size() < 4 || full_path.compare(full_path.size() - 4, 4, ".arh") != 0) {
                continue;
            }
            const std::string ard_path = full_path.substr(0, full_path.size() - 4) + ".ard";
            for (size_t j = 0; j < files.size(); ++j) {
                if (files[j]->GetFullPath() != ard_path) continue;
                const auto& ard_file = files[j];
                auto arh_entries = FileSys::TryEnumerateArhSubFiles(arh_file, ard_file->GetSize());
                if (arh_entries.empty()) break;
                LOG_INFO(Render_Vulkan,
                         "PreCacheShaders: '{}' recognized as an arh1 index for '{}' — {} "
                         "entries enumerated, replacing the single monolithic scan",
                         full_path, ard_path, arh_entries.size());
                for (const auto& entry : arh_entries) {
                    work_items.push_back(FileWorkItem{ard_file, std::nullopt, entry, std::nullopt, std::nullopt});
                }
                consumed_by_arh[j] = true; // Don't also queue a whole-file scan of the (huge) .ard file below.
                break;
            }
        }

        for (size_t i = 0; i < files.size(); ++i) {
            if (consumed_by_arh[i]) continue;
            const auto& file = files[i];
            auto arc_sub_files = FileSys::TryEnumerateArcSubFiles(file);
            if (!arc_sub_files.empty()) {
                LOG_INFO(Render_Vulkan,
                         "PreCacheShaders: '{}' recognized as an ARC archive — {} "
                         "sub-files enumerated, replacing the single monolithic scan",
                         file->GetFullPath(), arc_sub_files.size());
                for (const auto& sub : arc_sub_files) {
                    work_items.push_back(FileWorkItem{file, sub, std::nullopt, std::nullopt, std::nullopt});
                }
                continue;
            }
            // CRI Middleware CPK archives (One Piece: Pirate Warriors 3's
            // rom*.cpk files, and broadly across CRIWARE-licensed titles) —
            // a single self-contained file, unlike the .arh/.ard split
            // above, so it fits the same self-detecting expansion shape as
            // TryEnumerateArcSubFiles. See
            // docs/precache-scanner/FINDINGS.md section 8a.
            auto cpk_entries = FileSys::TryEnumerateCpkFiles(file);
            if (!cpk_entries.empty()) {
                LOG_INFO(Render_Vulkan,
                         "PreCacheShaders: '{}' recognized as a CPK archive — {} "
                         "entries enumerated, replacing the single monolithic scan",
                         file->GetFullPath(), cpk_entries.size());
                for (const auto& entry : cpk_entries) {
                    work_items.push_back(FileWorkItem{file, std::nullopt, std::nullopt, entry, std::nullopt});
                }
                continue;
            }
            // Metroid Prime Remastered's MaterialArchive.arc — another
            // single self-contained file. Only the real, Switch-native
            // "SNVN"-backend shader sources are returned (the "SDX "/DXBC
            // backend is a different instruction set entirely, out of
            // scope for this scanner — see
            // docs/precache-scanner/FINDINGS.md section 3).
            auto mpr_entries = FileSys::TryEnumerateMprSnvnShaderSources(file);
            if (!mpr_entries.empty()) {
                LOG_INFO(Render_Vulkan,
                         "PreCacheShaders: '{}' recognized as an MPR MaterialArchive — {} "
                         "SNVN shader sources enumerated, replacing the single monolithic scan",
                         file->GetFullPath(), mpr_entries.size());
                for (const auto& entry : mpr_entries) {
                    work_items.push_back(
                        FileWorkItem{file, std::nullopt, std::nullopt, std::nullopt, entry});
                }
                continue;
            }
            work_items.push_back(FileWorkItem{file, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
        }

        state->files_total.store(static_cast<int>(work_items.size()));
        LOG_INFO(Render_Vulkan, "PreCacheShaders: RomFS walk found {} files ({} scan units after ARC/ARH expansion)",
                 files.size(), work_items.size());

        VideoCommon::SpirvCache cache;
        cache.Load(spirv_path);

        const auto target = VideoCommon::LoadPrecacheCompilerTarget(target_path);
        if (!target) {
            state->error_message = "Vulkan compiler target disappeared before scanner start";
            return;
        }
        // Scanner final modules must use target copied from live Vulkan boot.
        // No generic/default profile exists on this path: unknown target means
        // no scan, not final SPIR-V compiled under invented capabilities.
        const Shader::Profile profile{target->profile};
        const Shader::HostTranslateInfo host_info{target->host_info};
        // Hash the decoded contract itself. Reconstructing only profile/host_info
        // here would silently reset the serialized revision triplet to defaults,
        // merging scanner entries across a recompiler/descriptor/specialization
        // ABI change even though the target file intentionally namespaces them.
        const u64 compiler_target_key = VideoCommon::CompilerTargetFingerprint(*target);
        LOG_INFO(Render_Vulkan,
                 "PreCacheShaders: using compiler target captured from this Vulkan boot: {:016x} "
                 "(GPU accuracy {})",
                 compiler_target_key, target->gpu_accuracy_mode);

        std::mutex seen_mutex;
        // Maps a raw blob PLUS its predecessor-output and descriptor context to
        // the state that a successful translation left for the next stage. A
        // blob-only set is unsound for BNSH: the same stage under a different
        // predecessor has different input declarations and descriptor numbers.
        ankerl::unordered_dense::map<u64, PrecacheStageResult> seen_contexts;

        const size_t nthreads = std::max(1u, std::thread::hardware_concurrency()-1u);
        Common::ThreadWorker workers{nthreads, "PreCacheShader"};

        for (const auto& item : work_items) {
            if (state->cancelled) break;
            workers.QueueWork([&, item]() {
                if (state->cancelled) return;
                ++state->files_processed;
                const auto& file = item.file; // Everything below already refers to `file`.
                // Whole-file read for every ordinary title (unchanged from
                // before this fix); a ranged read of just this one ARC
                // sub-file's already-known compressed extent, this one
                // arh-indexed entry's extent within the paired .ard file,
                // or this one CPK-indexed entry's extent, otherwise —
                // never the whole multi-GB container.
                auto raw = item.arc_range
                               ? file->ReadBytes(item.arc_range->comp_size, item.arc_range->offset)
                               : item.arh_range
                                     ? file->ReadBytes(item.arh_range->comp_size, item.arh_range->ard_offset)
                                     : item.cpk_range
                                           ? file->ReadBytes(item.cpk_range->file_size, item.cpk_range->file_offset)
                                           : item.mpr_range
                                                 ? file->ReadBytes(item.mpr_range->data_size,
                                                                    item.mpr_range->data_offset)
                                                 : file->ReadAllBytes();
                if (raw.size() < 4) return;

                // Xenoblade Chronicles 2's arh-indexed entries may themselves
                // be wrapped in Monolith Soft's own "xbc1" compression
                // container (zlib or zstd inside, per its own header field) —
                // a second, independent compression layer on top of the
                // arh/ard split itself. Unwrapped here, before the rest of
                // this function's existing zstd/Yaz0/SARC/pairtable/BNSH/
                // raw-SPH pipeline runs unmodified on the result — the same
                // "unwrap one known layer, then fall through to the generic
                // pipeline" pattern already used for every other format this
                // scanner supports. See docs/precache-scanner/FINDINGS.md
                // section 7.
                std::vector<u8> xbc1_storage;
                if (item.arh_range && Common::Compression::IsXBC1(raw)) {
                    xbc1_storage = Common::Compression::DecompressDataXBC1(raw);
                    if (xbc1_storage.empty()) return; // Malformed/unsupported — same as any other failed decompress.
                    raw = std::move(xbc1_storage);
                    if (raw.size() < 4) return;
                }

                // CPK entries may similarly be wrapped in CRI Middleware's own
                // "CRILAYLA" compression — see crilayla_compression.h for the
                // format itself and an important confidence caveat: unlike
                // every other decompressor in this scanner, this one could
                // not be tested against any real compressed sample (see
                // docs/precache-scanner/FINDINGS.md section 8a). Checked via
                // the entry's own extract_size > file_size (the conventional
                // CPK signal that an entry is compressed) as well as the
                // literal magic, since a corrupt/unusual entry might have one
                // but not the other.
                std::vector<u8> crilayla_storage;
                if (item.cpk_range && Common::Compression::IsCRILAYLA(raw)) {
                    crilayla_storage = Common::Compression::DecompressDataCRILAYLA(raw);
                    if (crilayla_storage.empty()) return; // Malformed/unsupported — same as any other failed decompress.
                    raw = std::move(crilayla_storage);
                    if (raw.size() < 4) return;
                }

                // MPR's SNVN-backend SShaderSource entries have their own
                // small header before a zlib stream: u32 (unused/unknown),
                // u32 decomp_size_hint, u8 flag, u32 compressed_size (13
                // bytes total). The "decomp_size_hint" field is NOT
                // trustworthy as an exact allocation size — confirmed
                // against real data during this investigation, where a
                // fixed-size decompress using that field's declared value
                // fails outright while the stream's real, larger output
                // decodes cleanly with a growable buffer — hence
                // DecompressDataZlib below ignoring it beyond a sizing
                // hint. See docs/precache-scanner/FINDINGS.md section 3.
                std::vector<u8> mpr_zlib_storage;
                if (item.mpr_range && raw.size() >= 13) {
                    u32 decomp_size_hint{}, compressed_size{};
                    std::memcpy(&decomp_size_hint, raw.data() + 4, 4);
                    std::memcpy(&compressed_size, raw.data() + 9, 4);
                    if (13 + static_cast<u64>(compressed_size) <= raw.size()) {
                        mpr_zlib_storage = Common::Compression::DecompressDataZlib(
                            std::span<const u8>(raw.data() + 13, compressed_size), decomp_size_hint);
                    }
                    if (mpr_zlib_storage.empty()) return; // Malformed/unsupported — same as any other failed decompress.
                    raw = std::move(mpr_zlib_storage);
                    if (raw.size() < 4) return;
                }

                // zstd frames start with a fixed 4-byte magic (0x28 0xB5 0x2F 0xFD).
                // TotK-style titles wrap most RomFS assets this way, frequently
                // against a shared dictionary (see dictionaries_by_id above) rather
                // than standalone. Decompress here, before anything else looks at
                // the bytes, so every check below transparently operates on the
                // real (uncompressed) shader-archive contents regardless of
                // whether the file on disk happened to be compressed.
                std::vector<u8> decompressed_storage;
                const u8* data = raw.data();
                size_t sz = raw.size();
                static constexpr u8 kZstdMagic[4] = {0x28, 0xB5, 0x2F, 0xFD};
                if (sz >= 4 && std::memcmp(raw.data(), kZstdMagic, 4) == 0) {
                    const auto dict_id = Common::Compression::GetZSTDFrameDictionaryID(raw);
                    if (!dict_id.has_value()) {
                        // Magic matched but the rest of the header doesn't parse as a
                        // real frame — corrupt or truncated. Nothing more to do.
                        ++state->zstd_failed;
                        return;
                    }
                    if (*dict_id == 0) {
                        decompressed_storage =
                            Common::Compression::DecompressDataZSTD(raw);
                    } else if (const auto it = dictionaries_by_id.find(*dict_id);
                               it != dictionaries_by_id.end()) {
                        decompressed_storage = Common::Compression::DecompressDataZSTDWithDictionary(
                            raw, it->second, /*max_decompressed_size=*/256ULL * 1024 * 1024);
                    } else {
                        // References a dictionary we don't have loaded — either this
                        // title has more dictionary-compressed resource types than the
                        // 3 known Pack/ZsDic.pack.zs entries cover, or ZsDic.pack.zs
                        // itself wasn't found/failed to parse earlier. Either way we
                        // can't decompress this file; skip it rather than guess.
                        ++state->zstd_failed;
                        return;
                    }
                    if (decompressed_storage.empty()) {
                        ++state->zstd_failed;
                        return;
                    }
                    ++state->zstd_decompressed;
                    data = decompressed_storage.data();
                    sz = decompressed_storage.size();
                } else if (Common::Compression::IsYaz0(raw)) {
                    // Yaz0 is Nintendo's older LZSS-family compression, still used by
                    // titles that predate (or otherwise don't use) TotK's zstd+
                    // dictionary scheme — e.g. Breath of the Wild and Super Mario
                    // Odyssey wrap their SARC-archived shader containers this way
                    // instead. Mutually exclusive with the zstd branch above: a file
                    // is one or the other, never both, at the top level.
                    decompressed_storage = Common::Compression::DecompressDataYaz0(raw);
                    if (decompressed_storage.empty()) {
                        ++state->zstd_failed; // No separate yaz0_failed counter; same
                                               // "compressed but couldn't decompress"
                                               // bucket as the zstd case.
                        return;
                    }
                    ++state->zstd_decompressed; // Counted alongside zstd successes —
                                                 // both mean "compression handled, real
                                                 // contents recovered."
                    data = decompressed_storage.data();
                    sz = decompressed_storage.size();
                } else if (sz >= 2 && raw[0] == 0x1f && raw[1] == 0x8b) {
                    // RFC 1952 gzip is a container-level compression layer, not a
                    // game rule. Once unwrapped, its contents flow through the same
                    // SARC/pairtable/BNSH/raw-SPH discovery below as every other
                    // already-readable scan unit.
                    decompressed_storage = Common::Compression::DecompressDataGzip(raw);
                    if (decompressed_storage.empty()) {
                        ++state->zstd_failed;
                        return;
                    }
                    ++state->zstd_decompressed;
                    data = decompressed_storage.data();
                    sz = decompressed_storage.size();
                }
                if (sz < 4) return;

                // Scanner runtime-state snapshot. Mirrors the fields
                // MakeRuntimeInfo() (vk_pipeline_cache.cpp) pulls from a real
                // previous_program when deriving a real pipeline's previous_stage_stores
                // -- see that function for the reference derivation this mirrors
                // field-for-field. Deliberately NOT holding onto the translated
                // Shader::IR::Program itself: its IR::Block/IR::Inst graph lives in
                // per-attempt ObjectPools (see try_translate_at below) that go out of
                // scope as soon as that lambda returns, so keeping a Program around
                // across sibling-stage calls would dangle. VaryingState (a bitset<512>
                // wrapper) and std::map<IR::Attribute,IR::Attribute> are both plain
                // value types with no pool dependency, so copying just these three
                // fields out avoids that trap entirely instead of trying to extend any
                // pool's lifetime.
                // Some titles (BOTW, SMO) package shader archives inside a SARC —
                // either the top-level file itself is a SARC (after the Yaz0/zstd
                // handling above), or — after this point resolves it — one of its
                // named entries needs its own, independent decompression before the
                // BNSH scan below can see real contents. Unlike the top-level
                // zstd/Yaz0 check, an entry's compression scheme isn't assumed to
                // match the outer container's: BOTW's outer Bootup_Graphics.pack is
                // an *uncompressed* SARC whose Shader/*.sbfsha entries are
                // individually Yaz0-compressed, while SMO's *.szs files are Yaz0
                // SARCs whose entries are stored plain. Each entry gets checked for
                // both schemes independently, same as a top-level file would be.
                //
                // Building this as a list of "scan units" rather than recursing
                // means the existing BNSH-scan/classify logic below (process_blob
                // onward) runs unmodified, once per unit — a plain file still
                // produces exactly one unit, so this is a no-op for every title
                // that doesn't use SARC-wrapped shaders.
                std::vector<ScanUnit> scan_units;
                std::optional<FileSys::SarcArchive> outer_sarc; // Entry spans point into this; must outlive scan_units' use of them.
                if (sz >= 4 && std::memcmp(data, "SARC", 4) == 0) {
                    outer_sarc = FileSys::SarcArchive::Parse(std::vector<u8>(data, data + sz));
                    if (outer_sarc->Ok()) {
                        for (const auto& entry : outer_sarc->Entries()) {
                            if (Common::Compression::IsYaz0(entry.data)) {
                                auto entry_decompressed =
                                    Common::Compression::DecompressDataYaz0(entry.data);
                                if (entry_decompressed.empty()) continue;
                                ScanUnit unit{std::move(entry_decompressed), nullptr, 0, entry.name};
                                unit.data = unit.owned.data();
                                unit.sz = unit.owned.size();
                                scan_units.push_back(std::move(unit));
                            } else if (entry.data.size() >= 4) {
                                scan_units.push_back(
                                    ScanUnit{{}, entry.data.data(), entry.data.size(), entry.name});
                            }
                        }
                    }
                }
                if (scan_units.empty() && sz >= 0x20 && data[0] == 0x55 && data[1] == 0xAA &&
                    data[2] == 0x38 && data[3] == 0x2D) {
                    // Nintendo's U8 archive is a documented big-endian tree of
                    // 12-byte nodes. It is independent of any particular game:
                    // a file node gives a byte range directly, so handing that
                    // range to the ordinary raw-NVN/BNSH scanner avoids a broad,
                    // error-prone scan of the complete archive.
                    const auto read_be_u32 = [&](size_t off, u32& out) {
                        if (off + 4 > sz) return false;
                        out = (static_cast<u32>(data[off]) << 24) |
                              (static_cast<u32>(data[off + 1]) << 16) |
                              (static_cast<u32>(data[off + 2]) << 8) |
                              static_cast<u32>(data[off + 3]);
                        return true;
                    };
                    u32 root_offset{};
                    if (read_be_u32(4, root_offset) &&
                        static_cast<size_t>(root_offset) + 12 <= sz) {
                        u32 root_kind_and_name{};
                        u32 node_count{};
                        if (read_be_u32(root_offset, root_kind_and_name) &&
                            read_be_u32(static_cast<size_t>(root_offset) + 8, node_count) &&
                            (root_kind_and_name >> 24) == 1u && node_count >= 1u &&
                            node_count <= (sz - root_offset) / 12) {
                            bool valid_tree = true;
                            std::vector<ScanUnit> u8_units;
                            u8_units.reserve(node_count > 0 ? node_count - 1 : 0);
                            for (u32 index = 1; index < node_count; ++index) {
                                const size_t node = static_cast<size_t>(root_offset) +
                                                    static_cast<size_t>(index) * 12;
                                u32 kind_and_name{};
                                u32 data_offset{};
                                u32 size_or_end{};
                                if (!read_be_u32(node, kind_and_name) ||
                                    !read_be_u32(node + 4, data_offset) ||
                                    !read_be_u32(node + 8, size_or_end)) {
                                    valid_tree = false;
                                    break;
                                }
                                const u32 kind = kind_and_name >> 24;
                                if (kind == 0) {
                                    if (static_cast<u64>(data_offset) + size_or_end > sz) {
                                        valid_tree = false;
                                        break;
                                    }
                                    if (size_or_end >= 4) {
                                        u8_units.push_back(ScanUnit{
                                            {}, data + data_offset, size_or_end,
                                            fmt::format("u8_entry_{}", index)});
                                    }
                                } else if (kind == 1) {
                                    // Directory nodes store their exclusive end
                                    // index, which must stay inside this tree.
                                    if (size_or_end <= index || size_or_end > node_count) {
                                        valid_tree = false;
                                        break;
                                    }
                                } else {
                                    valid_tree = false;
                                    break;
                                }
                            }
                            if (valid_tree && !u8_units.empty()) {
                                scan_units = std::move(u8_units);
                            }
                        }
                    }
                }
                if (scan_units.empty()) {
                    // Hyrule Warriors Definitive Edition (and, per the investigation
                    // this is based on, plausibly other Omega Force Switch titles)
                    // wraps shader-adjacent files in what this codebase calls a
                    // "pairtable": a u32 block count, followed by that many
                    // (u32 start_offset, u32 size) pairs describing byte ranges
                    // within this same buffer. Unlike SARC or Yaz0, there's no
                    // fixed magic byte sequence identifying this format — instead,
                    // detection relies on the table's internal self-consistency:
                    // in every real sample examined, each entry's start_offset
                    // lands at or just past the previous entry's end (a small
                    // alignment gap only), chaining across every single entry with
                    // zero exceptions — a pattern essentially impossible for
                    // unrelated binary data to produce by chance across many
                    // consecutive entries. See docs/precache-scanner/FINDINGS.md
                    // section 4 for the full investigation. Each unwrapped block is
                    // NOT compressed (confirmed via entropy analysis there) and does
                    // NOT wrap a BNSH container — it holds a variable-length
                    // reflection/metadata header (human-readable material parameter
                    // names, confirmed via embedded strings including the literal
                    // "ktglShaderConstants" — KTGL is Koei Tecmo's real internal
                    // engine name) followed by a bare, unwrapped Maxwell
                    // ProgramHeader at a per-block-varying offset — hence
                    // scan_full_for_raw_sph below, rather than assuming offset 0
                    // the way every other title's raw/bare-SPH case can.
                    if (sz >= 12) {
                        u32 num_blocks{};
                        std::memcpy(&num_blocks, data, 4);
                        // Sanity bound: every real sample seen has well under 200
                        // blocks; this also protects the loop below from an absurd
                        // count (corrupt data, or an unrelated file whose first 4
                        // bytes happen to form a large number) turning into an
                        // oversized allocation or iteration count.
                        constexpr u32 kMaxPlausibleBlocks = 100'000;
                        if (num_blocks >= 4 && num_blocks <= kMaxPlausibleBlocks &&
                            static_cast<u64>(num_blocks) * 8 + 4 <= sz) {
                            std::vector<std::pair<u32, u32>> entries; // (start_offset, size)
                            entries.reserve(num_blocks);
                            bool table_in_bounds = true;
                            for (u32 i = 0; i < num_blocks; ++i) {
                                u32 start{}, block_size{};
                                std::memcpy(&start, data + 4 + static_cast<size_t>(i) * 8, 4);
                                std::memcpy(&block_size, data + 4 + static_cast<size_t>(i) * 8 + 4, 4);
                                if (static_cast<u64>(start) + block_size > sz) {
                                    table_in_bounds = false;
                                    break;
                                }
                                entries.emplace_back(start, block_size);
                            }
                            if (table_in_bounds) {
                                size_t consistent = 0;
                                for (size_t i = 1; i < entries.size(); ++i) {
                                    const u64 prev_end =
                                        static_cast<u64>(entries[i - 1].first) + entries[i - 1].second;
                                    const u64 gap = entries[i].first >= prev_end
                                                         ? entries[i].first - prev_end
                                                         : std::numeric_limits<u64>::max();
                                    if (gap <= 8) ++consistent;
                                }
                                // Require the overwhelming majority of consecutive
                                // pairs to chain — every real table examined during
                                // this investigation chains 100% of the time; a
                                // small tolerance avoids rejecting a genuine table
                                // over one edge-case entry without meaningfully
                                // raising the false-positive rate on unrelated data.
                                if (consistent >= (entries.size() - 1) * 9 / 10) {
                                    for (const auto& [start, block_size] : entries) {
                                        if (block_size < 4) continue;
                                        ScanUnit unit{{}, data + start, block_size, {}, true};
                                        scan_units.push_back(std::move(unit));
                                    }
                                }
                            }
                        }
                    }
                }
                if (scan_units.empty()) {
                    // Not a recognized SARC (or it had no usable entries) — fall back
                    // to treating the whole (already zstd/Yaz0-decompressed, if
                    // applicable) buffer as a single unit, exactly as before this
                    // change existed.
                    scan_units.push_back(ScanUnit{{}, data, sz, {}});
                }

                for (const auto& scan_unit : scan_units) {
                const u8* data = scan_unit.data;
                size_t sz = scan_unit.sz;
                const std::string& entry_name = scan_unit.entry_name;
                if (sz < 4) continue;

                const auto process_blob = [&](const std::vector<u8>& blob, bool is_bnsh_derived,
                                              const PreviousStageStoresSnapshot* previous_stage,
                                              PreviousStageStoresSnapshot* out_stage_snapshot,
                                              Shader::Backend::Bindings starting_binding,
                                              Shader::Backend::Bindings* out_end_binding) -> bool {
                    // Claim one of a limited number of diagnostic-logging slots
                    // (thread-safe across the worker pool). diag_slot >= 0 means
                    // this call should log; this is temporary instrumentation to
                    // see exactly where real candidates succeed or fail, since
                    // shaders_found has stayed at 0 despite the extraction logic
                    // checking out correctly against a hand-verified real sample.
                    //
                    // Restricted to is_bnsh_derived candidates only: the previous
                    // run showed every one of the 20 slots consumed by raw_matched
                    // noise (identical common0_raw across every entry despite
                    // different sizes — a dead giveaway of unrelated small files
                    // that happen to share a leading byte coincidentally passing
                    // the loose per-file heuristic) before a single real
                    // BNSH-extracted candidate got a chance to log at all.
                    int diag_slot = -1;
                    if (is_bnsh_derived) {
                        int expected = state->diag_blobs_logged.load();
                        while (expected < kMaxDiagBlobs &&
                               !state->diag_blobs_logged.compare_exchange_weak(expected, expected + 1)) {
                        }
                        if (expected < kMaxDiagBlobs) diag_slot = expected;
                    }

                    if (blob.size() < sizeof(Shader::ProgramHeader)) {
                        if (diag_slot >= 0) {
                            LOG_INFO(Render_Vulkan,
                                     "PreCacheShaders diag[{}]: blob too small ({} bytes, need {})",
                                     diag_slot, blob.size(), sizeof(Shader::ProgramHeader));
                        }
                        return false;
                    }
                    Shader::ProgramHeader bsph{};
                    std::memcpy(&bsph, blob.data(), sizeof(bsph));

                    const auto has_valid_sph = [](const Shader::ProgramHeader& sph) noexcept {
                        const u32 type = sph.common0.shader_type.Value();
                        return sph.common0.version.Value() != 0 &&
                               sph.common0.sass_version.Value() != 0 && type >= 1 && type <= 5 &&
                               sph.common0.sph_type.Value() == ((type == 5) ? 2u : 1u);
                    };

                    // Validate SPH header fields before attempting to decode.
                    // Real NVIDIA shader program headers have:
                    //   version      (bits  5-9)  != 0  (typically 0x02)
                    //   shader_type  (bits 10-13) in [1,5]
                    //   sass_version (bits 17-20) != 0
                    // sph_type (bits 0-4) is NOT a flat constant — the header
                    // itself is a union of two different shapes depending on
                    // it: sph_type==1 selects the "vtg" layout (used by
                    // vertex/tess-control/tess-eval/geometry — shader_type
                    // 1-4), while sph_type==2 selects the "ps" (pixel/
                    // fragment) layout, used only by shader_type==5. Treating
                    // sph_type==1 as universally required was silently
                    // rejecting every real fragment shader in the ROM.
                    // Files whose bytes happen to pass only the shader_type
                    // check (asset data, textures, etc.) are rejected here to
                    // avoid flooding the Maxwell decoder with garbage and
                    // spamming "Invalid insn" assertions.
                    u32 common0_raw{};
                    std::memcpy(&common0_raw, blob.data(), sizeof(common0_raw));
                    const u32 sph_type    = (common0_raw >>  0) & 0x1Fu;
                    const u32 version     = (common0_raw >>  5) & 0x1Fu;
                    const u32 shader_type = (common0_raw >> 10) & 0x0Fu;
                    const u32 sass_ver    = (common0_raw >> 17) & 0x0Fu;
                    if (diag_slot >= 0) {
                        LOG_INFO(Render_Vulkan,
                                 "PreCacheShaders diag[{}]: blob_size={} common0_raw={:08x} "
                                 "sph_type={} version={} shader_type={} sass_ver={}",
                                 diag_slot, blob.size(), common0_raw, sph_type, version,
                                 shader_type, sass_ver);
                    }
                    if (!has_valid_sph(bsph)) {
                        if (diag_slot >= 0) {
                            LOG_INFO(Render_Vulkan,
                                     "PreCacheShaders diag[{}]: rejected — invalid SPH signature",
                                     diag_slot);
                        }
                        return false;
                    }

                    // Require at least one complete instruction beyond the header.
                    // A raw NVN program can be embedded within a larger archive
                    // entry, where unrelated trailing metadata makes the remaining
                    // entry length non-8-aligned. That must not invalidate the
                    // header-led program; discard only the incomplete tail word.
                    const size_t available_payload = blob.size() - sizeof(Shader::ProgramHeader);
                    const size_t payload = available_payload & ~(sizeof(u64) - 1);
                    if (payload < sizeof(u64)) {
                        if (diag_slot >= 0) {
                            LOG_INFO(Render_Vulkan,
                                     "PreCacheShaders diag[{}]: rejected — payload={} has no complete "
                                     "instruction", diag_slot, available_payload);
                        }
                        return false;
                    }

                    // NOTE: a standalone "decode the first instruction in isolation"
                    // pre-check used to live here. It was removed after tracing the
                    // real cause of decode failures to CFG's start address, which the
                    // live pipeline computes as env.StartAddress() + sizeof(SPH) —
                    // StartAddress() being live GPU-register-supplied context that a
                    // static file scan cannot know. See process_stage_offset's
                    // ControlCode-fallback comment for how this is being worked
                    // around instead.
                    if (diag_slot >= 0) {
                        u64 first_insn{};
                        std::memcpy(&first_insn, blob.data() + sizeof(Shader::ProgramHeader), 8);
                        LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: first_insn={:016x}, proceeding to CFG/translate",
                                 diag_slot, first_insn);
                    }

                    const u32 t = shader_type;

                    // Deduplicate only equal complete contexts. This permits a
                    // repeated BNSH stage after a different predecessor to make a
                    // distinct speculative entry, while avoiding repeated work for
                    // truly identical BNSH variations and raw standalone sources.
                    u64 context_key = Common::CityHash64(
                        reinterpret_cast<const char*>(blob.data()), blob.size());
                    const auto combine_context = [&context_key](u64 value) {
                        context_key ^= value + 0x9e3779b97f4a7c15ULL +
                            (context_key << 6) + (context_key >> 2);
                    };
                    combine_context(VideoCommon::ComputeBindingKey(starting_binding));
                    if (previous_stage) {
                        combine_context(std::hash<std::bitset<512>>{}(
                            previous_stage->stores.mask));
                        combine_context(std::hash<std::bitset<512>>{}(
                            previous_stage->passthrough.mask));
                        combine_context(previous_stage->is_geometry_passthrough);
                        for (const auto& [from, to] : previous_stage->legacy_stores_mapping) {
                            combine_context((static_cast<u64>(from) << 32) |
                                            static_cast<u64>(to));
                        }
                    } else {
                        combine_context(~0ULL);
                    }
                    {
                        std::lock_guard g{seen_mutex};
                        const auto it = seen_contexts.find(context_key);
                        if (it != seen_contexts.end()) {
                            if (out_stage_snapshot) {
                                *out_stage_snapshot = it->second.stage_snapshot;
                            }
                            if (out_end_binding) {
                                *out_end_binding = it->second.end_binding;
                            }
                            return true;
                        }
                    }

                    const Shader::Stage stage = [t]()->Shader::Stage {
                        switch(t){case 1:return Shader::Stage::VertexB;
                                  case 2:return Shader::Stage::TessellationControl;
                                  case 3:return Shader::Stage::TessellationEval;
                                  case 4:return Shader::Stage::Geometry;
                                  default:return Shader::Stage::Fragment;}
                    }();
                    // code[] now includes the SPH as its first sizeof(ProgramHeader)/8
                    // words, followed by the payload — matching what a live
                    // GraphicsEnvironment's code[] always contains (it's read starting
                    // at start_address, which IS the SPH's own address there). The
                    // scanner varies that address's alignment below, while keeping the
                    // byte span itself unchanged so CalculateHash() agrees with
                    // GenericEnvironment::Analyze() on the same shader.
                    std::vector<u64> code(sizeof(Shader::ProgramHeader) / 8 + payload / 8);
                    std::memcpy(code.data(), blob.data(), sizeof(Shader::ProgramHeader));
                    std::memcpy(code.data() + sizeof(Shader::ProgramHeader) / 8,
                                blob.data() + sizeof(Shader::ProgramHeader), payload);
                    const u32 lm = static_cast<u32>(bsph.LocalMemorySize()) +
                                   static_cast<u32>(bsph.common3.shader_local_memory_crs_size);
                    if (diag_slot >= 0) {
                        LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: entering CFG/translate, stage={}",
                                 diag_slot, static_cast<int>(stage));
                    }

                    // A BNSH/raw-NVN program is serialized from its ProgramHeader
                    // onward. The missing fact in a static file is not a later entry
                    // offset: it is the GPU address at which that header is placed.
                    // Maxwell inserts a scheduler word at each 32-byte instruction
                    // group, so that address changes which words CFG treats as
                    // schedulers. Try the four possible 8-byte alignments while
                    // keeping the program and its identity intact.
                    constexpr size_t kMaxLoggedCandidateFailures = 2;
                    size_t logged_candidate_failures{};
                    const auto try_translate_at = [&](u32 program_start_address,
                                                      bool artifact_only) -> bool {
                        try {
                            VideoCommon::SpeculativeShaderEnvironment env{
                                code, program_start_address, stage, lm, 0,
                                std::array<u32, 3>{1u, 1u, 1u}, 1u, bsph,
                                /*code_offset_in_program=*/0u};
                            // The scan can enumerate scheduler alignments, but
                            // a static file does not prove which GPU address the
                            // runtime will use. Keep every result out of exact
                            // publication (and the current post-CFG template
                            // boundary) until a deeper pre-CFG artifact exists.
                            env.MarkSchedulingAlignmentUnknown();
                            Shader::ObjectPool<Shader::Maxwell::Flow::Block> fp(16);
                            Shader::ObjectPool<Shader::IR::Inst> ip(8192);
                            Shader::ObjectPool<Shader::IR::Block> bp(32);
                            const auto cfg_begin = std::chrono::steady_clock::now();
                            Shader::Maxwell::Flow::CFG cfg(
                                env, fp, program_start_address +
                                             static_cast<u32>(sizeof(Shader::ProgramHeader)), false);
                            state->cfg_us.fetch_add(static_cast<u64>(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - cfg_begin).count()));
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan,
                                         "PreCacheShaders diag[{}]: CFG construction OK at program alignment {}",
                                         diag_slot, program_start_address);
                            }

                            // This is the same two-path identity calculation the live shader
                            // cache uses: SPH-to-self-branch when the fast path can identify a
                            // terminal branch, otherwise the complete CFG-read span. The latter
                            // matters for legitimate BNSH programs that end without either of
                            // the two historical self-branch encodings.
                            const u64 unique_hash = env.CalculateHash();
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: unique_hash={:016x}",
                                         diag_slot, unique_hash);
                            }
                            if (!artifact_only) {
                                ++state->shaders_found;
                            }

                            // Persist CFG artifacts with known control flow.
                            ++state->cfg_artifact_candidates;
                            // CFG construction observes cbuf-backed control flow. Scheduler
                            // placement is captured in the artifact key; later lowering state is
                            // checked by fresh-module validation, not this CFG-only gate.
                            if (env.HasUnknownCbufDependencies()) {
                                ++state->cfg_artifact_cbuf_rejections;
                            } else {
                                try {
                                    auto artifact_cfg = cfg.MakeTemplate();
                                    const bool has_indirect_branch = std::any_of(
                                        artifact_cfg.blocks.begin(), artifact_cfg.blocks.end(),
                                        [](const auto& block) {
                                            return !block.indirect_branches.empty();
                                        });
                                    const auto artifact_key =
                                        artifact_cfg.functions.empty()
                                            ? std::optional<VideoCommon::PrecacheCfgArtifactKey>{}
                                            : VideoCommon::MakePrecacheCfgArtifactKey(
                                                  unique_hash, stage,
                                                  artifact_cfg.functions.front().entrypoint.Offset());
                                    if (has_indirect_branch || !artifact_key) {
                                        ++state->cfg_artifact_branches_rejections;
                                    } else {
                                        std::vector<Shader::Maxwell::PredecodedInstruction>
                                            decoded_instructions;
                                        for (const auto& block : artifact_cfg.blocks) {
                                            for (auto location = block.begin; location != block.end;
                                                 ++location) {
                                                const u64 instruction{
                                                    env.ReadInstruction(location.Offset())};
                                                decoded_instructions.push_back(
                                                    {.location = location.Offset(),
                                                     .instruction = instruction,
                                                     .opcode = Shader::Maxwell::Decode(instruction)});
                                            }
                                        }
                                        std::ranges::sort(decoded_instructions, {},
                                                          &Shader::Maxwell::PredecodedInstruction::location);
                                        const auto duplicate = std::adjacent_find(
                                            decoded_instructions.begin(), decoded_instructions.end(),
                                            [](const auto& lhs, const auto& rhs) {
                                                return lhs.location == rhs.location;
                                            });
                                        if (decoded_instructions.empty() ||
                                            duplicate != decoded_instructions.end()) {
                                            ++state->cfg_artifact_branches_rejections;
                                        } else {
                                            VideoCommon::PrecacheCfgArtifact artifact{
                                                .key = *artifact_key,
                                                .cfg = std::move(artifact_cfg),
                                                .decoded_instructions = std::move(decoded_instructions),
                                            };
                                            if (artifact.IsValid()) {
                                                std::lock_guard lock{state->cfg_artifact_mutex};
                                                if (state->cfg_artifact_keys.insert(artifact.key).second) {
                                                    if (state->cfg_artifacts.size() <
                                                        kMaxPersistedCfgArtifacts) {
                                                        ++state->cfg_artifact_stages[
                                                            static_cast<size_t>(artifact.key.stage)];
                                                        state->cfg_artifacts.push_back(std::move(artifact));
                                                        ++state->cfg_artifact_stored;
                                                    } else {
                                                        ++state->cfg_artifact_capacity_rejections;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } catch (...) {
                                    // Keep scanner translation usable on artifact failure.
                                    ++state->cfg_artifact_branches_rejections;
                                }
                            }

                            // Later scheduler alignments add CFG artifacts only.
                            if (artifact_only) {
                                return true;
                            }

                            // CFG discovery may read cbuf-backed branch data and the scanner
                            // itself records that this alignment was synthetic. Those facts
                            // remain disqualifying for a final SPIR-V module, but they do not
                            // answer whether the following frontend boundary consumed unknown
                            // runtime state. Snapshot them, measure BuildProgramTemplate alone,
                            // then restore them before finalization/final-module validation.
                            const auto cfg_dependencies = env.TakeDependencySnapshot();
                            const auto template_begin = std::chrono::steady_clock::now();
                            auto prog = Shader::Maxwell::BuildProgramTemplate(ip, bp, env, cfg,
                                                                                host_info);
                            const auto& frontend_dependencies{prog.frontend_dependencies};
                            const auto uses = [&frontend_dependencies](
                                                 Shader::IR::FrontendDependency dependency) {
                                return frontend_dependencies.Uses(dependency);
                            };
                            if (uses(Shader::IR::FrontendDependency::ConstantBuffer) ||
                                uses(Shader::IR::FrontendDependency::ConstantBufferSize) ||
                                uses(Shader::IR::FrontendDependency::ConstantBufferReplacement)) {
                                ++state->frontend_manifest_cbuf;
                            }
                            if (uses(Shader::IR::FrontendDependency::TextureType) ||
                                uses(Shader::IR::FrontendDependency::TexturePixelFormat) ||
                                uses(Shader::IR::FrontendDependency::TextureIntegerFormat) ||
                                uses(Shader::IR::FrontendDependency::TextureBinding) ||
                                uses(Shader::IR::FrontendDependency::ProprietaryDriver)) {
                                ++state->frontend_manifest_resource;
                            }
                            if (uses(Shader::IR::FrontendDependency::ViewportTransform)) {
                                ++state->frontend_manifest_viewport;
                            }
                            if (uses(Shader::IR::FrontendDependency::ComputeLaunch)) {
                                ++state->frontend_manifest_compute;
                            }
                            if (uses(Shader::IR::FrontendDependency::HLEMacro)) {
                                ++state->frontend_manifest_hle;
                            }
                            if (uses(Shader::IR::FrontendDependency::GeometryPassthrough)) {
                                ++state->frontend_manifest_interface;
                            }
                            state->template_us.fetch_add(static_cast<u64>(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - template_begin).count()));
                            // Snapshot dependencies exactly at proposed template
                            // boundary. FinalizeProgramTemplate() is allowed to
                            // consume real cbuf/resource/interface state later;
                            // consuming it here would make scanner reuse unsound.
                            const bool template_unknown_cbuf = env.HasUnknownCbufDependencies();
                            const bool template_unknown_texture =
                                env.HasUnknownTextureDependencies();
                            const bool template_unknown_viewport =
                                env.HasUnknownViewportDependency();
                            const bool template_unknown_compute =
                                env.HasUnknownComputeLaunchDependency();
                            const bool template_unknown_hle_macro =
                                env.HasUnknownHLEMacroDependency();
                            const bool template_unknown_interface =
                                env.HasUnknownInterfaceDependency();
                            const bool template_unknown_scheduling_alignment =
                                env.HasUnknownSchedulingAlignmentDependency();
                            if (env.HasStateIndependentTemplateContract()) {
                                ++state->template_boundary_eligible;
                                ++state->frontend_artifact_candidates;
                                VideoCommon::PrecacheFrontendFreezeError freeze_error;
                                auto frozen = VideoCommon::FreezePrecacheFrontendArtifact(
                                    prog, freeze_error);
                                if (!frozen) {
                                    switch (freeze_error) {
                                    case VideoCommon::PrecacheFrontendFreezeError::UnsupportedInfo:
                                        ++state->frontend_artifact_info_rejections;
                                        break;
                                    case VideoCommon::PrecacheFrontendFreezeError::AssociatedPseudoOperation:
                                        ++state->frontend_artifact_pseudo_rejections;
                                        break;
                                    case VideoCommon::PrecacheFrontendFreezeError::UnsupportedValue:
                                        ++state->frontend_artifact_value_rejections;
                                        break;
                                    case VideoCommon::PrecacheFrontendFreezeError::InvalidGraph:
                                    case VideoCommon::PrecacheFrontendFreezeError::TooLarge:
                                        ++state->frontend_artifact_graph_rejections;
                                        break;
                                    case VideoCommon::PrecacheFrontendFreezeError::None:
                                        break;
                                    }
                                } else {
                                    VideoCommon::PrecacheFrontendArtifactRecord record{
                                        .key = {
                                            .program_identity = unique_hash,
                                            .compiler_target_fingerprint = compiler_target_key,
                                            .source_header = Common::CityHash64(
                                                reinterpret_cast<const char*>(&bsph), sizeof(bsph)),
                                            .local_memory_size = prog.local_memory_size,
                                            .stage = stage,
                                            .scheduler_slot = static_cast<u8>(
                                                program_start_address % 32),
                                            .exits_to_dispatcher = false,
                                        },
                                        .artifact = std::move(*frozen),
                                    };
                                    if (!record.IsValid()) {
                                        ++state->frontend_artifact_graph_rejections;
                                    } else {
                                        std::lock_guard lock{state->frontend_artifact_mutex};
                                        if (state->frontend_artifact_keys.insert(record.key).second) {
                                            if (state->frontend_artifacts.size() <
                                                kMaxPersistedFrontendArtifacts) {
                                                state->frontend_artifacts.push_back(std::move(record));
                                                ++state->frontend_artifact_stages[
                                                    static_cast<size_t>(stage)];
                                                ++state->frontend_artifact_stored;
                                            } else {
                                                ++state->frontend_artifact_capacity_rejections;
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (template_unknown_cbuf) {
                                    ++state->template_boundary_cbuf_rejections;
                                }
                                if (template_unknown_texture) {
                                    ++state->template_boundary_texture_rejections;
                                }
                                if (template_unknown_viewport) {
                                    ++state->template_boundary_viewport_rejections;
                                }
                                if (template_unknown_compute) {
                                    ++state->template_boundary_compute_rejections;
                                }
                                if (template_unknown_hle_macro) {
                                    ++state->template_boundary_hle_macro_rejections;
                                }
                                if (template_unknown_interface) {
                                    ++state->template_boundary_interface_rejections;
                                }
                                if (template_unknown_scheduling_alignment) {
                                    ++state->template_boundary_scheduling_alignment_rejections;
                                }
                            }
                            env.MergeDependencySnapshot(cfg_dependencies);
                            const auto finalize_begin = std::chrono::steady_clock::now();
                            Shader::Maxwell::FinalizeProgramTemplate(prog, env, host_info);
                            state->finalize_us.fetch_add(static_cast<u64>(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - finalize_begin).count()));
                            const bool has_unknown_cbuf = env.HasUnknownCbufDependencies();
                            const bool has_unknown_texture = env.HasUnknownTextureDependencies();
                            const bool has_unknown_viewport = env.HasUnknownViewportDependency();
                            const bool has_unknown_compute =
                                env.HasUnknownComputeLaunchDependency();
                            const bool has_unknown_hle_macro = env.HasUnknownHLEMacroDependency();
                            const bool has_unknown_interface =
                                env.HasUnknownInterfaceDependency();
                            const bool has_unknown_scheduling_alignment =
                                env.HasUnknownSchedulingAlignmentDependency();
                            if (has_unknown_cbuf) {
                                ++state->unknown_cbuf_candidates;
                            }
                            if (has_unknown_texture) {
                                ++state->unknown_texture_candidates;
                            }
                            if (has_unknown_viewport) {
                                ++state->unknown_viewport_candidates;
                            }
                            if (has_unknown_compute) {
                                ++state->unknown_compute_candidates;
                            }
                            if (has_unknown_hle_macro) {
                                ++state->unknown_hle_macro_candidates;
                            }
                            if (has_unknown_interface) {
                                ++state->unknown_interface_candidates;
                            }
                            if (has_unknown_scheduling_alignment) {
                                ++state->unknown_scheduling_alignment_candidates;
                            }
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan,
                                         "PreCacheShaders diag[{}]: TranslateProgram OK at program alignment {}",
                                         diag_slot, program_start_address);
                            }
                            Shader::Backend::Bindings binding = starting_binding;
                            Shader::RuntimeInfo rt{};
                            // Scanner runtime-state snapshot: for every stage except VertexB,
                            // previous_stage_stores is a real, IS-read field of
                            // SpirvRelevantHash(stage) (see that function's own comment)
                            // -- and for a REAL draw it reflects the actual preceding
                            // stage's real output layout, never the "no previous
                            // program" sentinel MakeRuntimeInfo only falls back to when
                            // there genuinely is no previous stage. previous_stage, when
                            // non-null, is exactly that real data -- captured from
                            // translating this same BNSH shader program's earlier stage
                            // moments ago (see process_bnsh_at's stage loop) -- not a
                            // guess, the same field access MakeRuntimeInfo itself uses.
                            if (stage != Shader::Stage::VertexB && previous_stage) {
                                rt.previous_stage_stores = previous_stage->stores;
                                rt.previous_stage_legacy_stores_mapping =
                                    previous_stage->legacy_stores_mapping;
                                if (previous_stage->is_geometry_passthrough) {
                                    rt.previous_stage_stores.mask |= previous_stage->passthrough.mask;
                                }
                            } else {
                                // VertexB never has a previous program in a real pipeline
                                // either (see SpirvRelevantHash's own comment on this,
                                // which is also why VertexB skips this field in the hash
                                // entirely regardless of what's set here) -- and if this
                                // stage DOES read the field but no real sibling data is
                                // available (this program doesn't declare an earlier
                                // stage, or that stage failed to translate), honesty
                                // beats a wrong guess: fall back to the same
                                // conservative "no restriction" sentinel MakeRuntimeInfo
                                // itself uses in the equivalent real case, rather than
                                // inventing something with no basis.
                                rt.previous_stage_stores.mask.set();
                            }
                            // Triangles are the overwhelmingly common input topology.
                            // Geometry and tessellation shaders that need a different
                            // topology will be recompiled correctly during live play.
                            rt.input_topology = Shader::InputTopology::Triangles;
                            if (stage == Shader::Stage::Fragment) {
                                // Accept all frag color output types conservatively.
                                rt.frag_color_types.fill(Shader::FragmentOutputType::Float);
                            }
                            // Deliberate runtime defaults (runtime_info.h) for
                            // the fields this speculative path has no real per-draw signal
                            // for.
                            rt.ApplySpeculativeDefaults(stage, prog.info);
                            Shader::Maxwell::ConvertLegacyToGeneric(prog, rt);
                            const auto emit_begin = std::chrono::steady_clock::now();
                            auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile,rt,prog,binding);
                            state->emit_us.fetch_add(static_cast<u64>(
                                std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - emit_begin).count()));
                            // The scanner supplies synthetic descriptor handles. Match real
                            // execution by keying complete observations on the cbuf slot that
                            // produced each handle; retain the raw key for any untracked query
                            // or while handle-specific polymorphism is active.
                            const bool use_logical_texture_key =
                                !VideoCommon::HasActivePhase4LogicalTextureSlot(
                                    env.CapturedLogicalTextureSlots()) &&
                                VideoCommon::HasCompleteLogicalTextureCoverage(
                                    env.CapturedLogicalTextureSlots(),
                                    env.CapturedLogicalTextureHandles(),
                                    env.CapturedTextureTypes(), env.CapturedTexturePixelFormats());
                            const u64 texture_key =
                                use_logical_texture_key
                                    ? VideoCommon::ComputeLogicalTextureKey(
                                          env.CapturedLogicalTextureSlots())
                                    : VideoCommon::ComputeTextureKey(env.CapturedTextureTypes(),
                                                                     env.CapturedTexturePixelFormats());
                            cache.RecordTextureKeyMode(use_logical_texture_key);
                            // Every real Lookup()/Insert() call folds viewport_transform_state
                            // (VertexB only) and binding_key into runtime_key before touching
                            // the cache (see FoldViewportTransformState/FoldBindingKey's doc
                            // comment in spirv_cache.h) — passing rt.Hash() straight through
                            // here, unfolded, put every entry this scanner ever inserted in a
                            // format that could never match a real one, regardless of how
                            // accurate any other part of the guess was.
                            //
                            // SpirvRelevantHash(stage), not Hash(): folds only the fields this
                            // stage's SPIR-V emission actually reads (runtime_info.h) instead
                            // of the whole struct. Directly relevant to the two guesses just
                            // above — rt.input_topology=Triangles only ever mattered for
                            // Geometry anyway (never read for Fragment/VertexB codegen), so
                            // real Fragment/VertexB draws using Lines/Points no longer need to
                            // coincidentally match a guess that was never going to affect their
                            // actual SPIR-V in the first place.
                            u64 runtime_key = stage == Shader::Stage::Compute
                                                  ? VideoCommon::ComputeWorkgroupKey(
                                                        /*shared_memory_size=*/0, {1u, 1u, 1u})
                                                  : rt.SpirvRelevantHash(stage);
                            if (stage == Shader::Stage::VertexB) {
                                runtime_key = VideoCommon::FoldViewportTransformState(
                                    runtime_key, env.ReadViewportTransformState());
                            }
                            // Same diagnostic split as the live speculative path in
                            // vk_pipeline_cache.cpp: capture the pre-binding-fold "core"
                            // component so a later stale miss against this scanner-inserted
                            // entry can be attributed to the core RuntimeInfo state vs. the
                            // binding-offset guess specifically, instead of only ever seeing
                            // an opaque "runtime differs" on the folded key. See
                            // spirv_cache.h's Insert()/InsertSpeculative() doc comments.
                            const u64 diag_base_runtime_hash = runtime_key;
                            const auto diag_runtime_fields = rt.SpirvRelevantFieldHashes(stage);
                            const u64 diag_binding_key = stage == Shader::Stage::Compute
                                                             ? 0
                                                             : VideoCommon::ComputeBindingKey(starting_binding);
                            if (stage != Shader::Stage::Compute) {
                                runtime_key = VideoCommon::FoldBindingKey(runtime_key,
                                                                          diag_binding_key);
                            }
                            // Scanner defaults are unknown state, never observed
                            // values. Any unknown cbuf, texture shape, viewport, compute,
                            // HLE, or geometry-interface dependency can alter lowering or
                            // metadata, so none may publish an exact final module.
                            // Translation still runs for template work, stage-chain
                            // diagnostics, and timing.
                            if (env.HasCompleteFinalModuleContract() && unique_hash != 0 &&
                                !spirv.empty()) {
                                cache.InsertSpeculative(
                                    unique_hash, stage, compiler_target_key, runtime_key,
                                    texture_key, std::move(spirv), binding,
                                    diag_base_runtime_hash, diag_binding_key,
                                    diag_runtime_fields,
                                    VideoCommon::SpirvCacheEntrySource::PrecacheScanner);
                                ++state->final_entries;
                            } else {
                                if (has_unknown_cbuf) {
                                    ++state->final_cbuf_contract_rejections;
                                }
                                if (has_unknown_texture) {
                                    ++state->final_texture_contract_rejections;
                                }
                                if (has_unknown_viewport) {
                                    ++state->final_viewport_contract_rejections;
                                }
                                if (has_unknown_compute) {
                                    ++state->final_compute_contract_rejections;
                                }
                                if (has_unknown_hle_macro) {
                                    ++state->final_hle_macro_contract_rejections;
                                }
                                if (has_unknown_interface) {
                                    ++state->final_interface_contract_rejections;
                                }
                                if (has_unknown_scheduling_alignment) {
                                    ++state->final_scheduling_alignment_contract_rejections;
                                }
                                if (unique_hash == 0 || spirv.empty()) {
                                    ++state->final_identity_contract_rejections;
                                }
                            }

                            ++state->shaders_translated;
                            if (program_start_address != 0) {
                                ++state->alignment_translated_entries;
                            }
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan,
                                         "PreCacheShaders diag[{}]: fully translated OK at program alignment {}",
                                         diag_slot, program_start_address);
                            }
                            // Written only here, on confirmed full success (translate +
                            // emit + insert all completed) -- not right after
                            // TranslateProgram above, so a later failure/exception in
                            // this same attempt can never leave the caller thinking this
                            // stage succeeded when process_blob is about to return false.
                            const PrecacheStageResult result{
                                {prog.info.stores, prog.info.legacy_stores_mapping,
                                 prog.info.passthrough, prog.is_geometry_passthrough}, binding};
                            {
                                std::lock_guard g{seen_mutex};
                                seen_contexts.insert_or_assign(context_key, result);
                            }
                            if (out_stage_snapshot) {
                                *out_stage_snapshot = result.stage_snapshot;
                            }
                            if (out_end_binding) {
                                *out_end_binding = result.end_binding;
                            }
                            // A second VertexB translate guessing viewport_transform_state=0
                            // used to run here (mirroring PipelineCache::SubmitSpeculativeShader).
                            // Measured across three full TotK sessions with 2000+ speculative
                            // entries sitting in the cache: zero hits against any of them,
                            // scanner or live-path alike. cbuf_key — hardcoded to 0 for every
                            // speculative entry, never guessed — is what actually gates a hit
                            // (see FoldViewportTransformState's doc comment in spirv_cache.h),
                            // and it's off the table for real shaders that specialize on cbuf
                            // content, which is most of them. Guessing harder on
                            // viewport_transform_state doesn't move that ceiling, so removed
                            // rather than doubling down on it — see PipelineCache::
                            // SubmitSpeculativeShader for the matching removal on the live path.
                            return true;
                        } catch (const std::exception& e) {
                            if (diag_slot >= 0 &&
                                logged_candidate_failures++ < kMaxLoggedCandidateFailures) {
                                LOG_INFO(Render_Vulkan,
                                         "PreCacheShaders diag[{}]: program alignment {} threw std::exception: {}",
                                         diag_slot, program_start_address, e.what());
                            }
                            return false;
                        } catch (...) {
                            if (diag_slot >= 0 &&
                                logged_candidate_failures++ < kMaxLoggedCandidateFailures) {
                                LOG_INFO(Render_Vulkan,
                                         "PreCacheShaders diag[{}]: program alignment {} threw unknown exception",
                                         diag_slot, program_start_address);
                            }
                            return false;
                        }
                    };

                    bool full_translation_succeeded{};
                    for (const u32 alignment : {0u, 8u, 16u, 24u}) {
                        if (try_translate_at(alignment, full_translation_succeeded)) {
                            full_translation_succeeded = true;
                        }
                    }
                    if (full_translation_succeeded) {
                        return true;
                    }
                    ++state->shaders_failed;
                    return false;
                };

                const auto log_unrecognized_sample = [&]() {
                    std::lock_guard g{state->sample_mutex};
                    if (state->samples_logged >= kMaxSamples) return;
                    ++state->samples_logged;
                    const size_t dump_len = std::min<size_t>(sz, 16);
                    std::string hex;
                    for (size_t i = 0; i < dump_len; ++i) {
                        hex += fmt::format("{:02x} ", data[i]);
                    }
                    LOG_INFO(Render_Vulkan,
                             "PreCacheShaders: unrecognized file '{}'{}{}{}{}{} ({} bytes), first {} bytes: {}",
                             file->GetFullPath(),
                             entry_name.empty() ? "" : fmt::format(" [SARC entry '{}']", entry_name),
                             item.arc_range
                                 ? fmt::format(" [ARC sub-file @ offset {}]", item.arc_range->offset)
                                 : "",
                             item.arh_range
                                 ? fmt::format(" [arh entry @ ard offset {}]", item.arh_range->ard_offset)
                                 : "",
                             item.cpk_range
                                 ? fmt::format(" [CPK entry '{}/{}']", item.cpk_range->dir_name,
                                               item.cpk_range->file_name)
                                 : "",
                             item.mpr_range
                                 ? fmt::format(" [MPR SNVN entry @ offset {}]", item.mpr_range->data_offset)
                                 : "",
                             sz, dump_len, hex);
                };

                constexpr u32 BNSH_MAGIC = 0x48534E42u; // "BNSH"

                // Extracts and translates every shader from a BNSH blob starting
                // at byte offset `base` within data/sz. This is the real BNSH
                // (BFRES-family) container format, verified field-by-field
                // against an actual decompressed TotK BFSHA sample rather than
                // reconstructed from a generic guess:
                //
                //   base + 0x00  BinaryHeader (32 bytes; shared with other
                //                 BFRES-family formats). Field of interest:
                //                 BlockOffset (u16 @ 0x16) — where the
                //                 format-specific header starts (conventionally
                //                 96, but read rather than assumed).
                //   base + BlockOffset
                //                GRSC header (56 bytes). Fields of interest:
                //                 magic (u32 @ 0x00, == "grsc", lowercase —
                //                 despite the reference implementation's own
                //                 comment calling it "GRSC"), NumVariation
                //                 (u32 @ 0x1C), VariationStartOffset (u64 @ 0x20).
                //   base + VariationStartOffset + i*64
                //                One VariationHeader per variation (64 bytes
                //                each). Field of interest: BinaryOffset
                //                (u64 @ 0x10).
                //   base + BinaryOffset
                //                BnshShaderProgramHeader (176 bytes). Six u64
                //                per-stage offsets at fixed positions: Vertex
                //                @0x08, TessControl @0x10, TessEval @0x18,
                //                Geometry @0x20, Fragment @0x28, Compute @0x30.
                //                Zero means that stage isn't present.
                //   base + <stage offset>
                //                ShaderCode header (64 bytes): 8 bytes unused,
                //                then ControlCodeOffset (u64 @0x08), ByteCodeOffset
                //                (u64 @0x10), ByteCodeSize (u32 @0x18),
                //                ControlCodeSize (u32 @0x1C), 32 bytes reserved.
                //   base + ByteCodeOffset
                //                The actual per-stage data. The first 48 bytes
                //                here are some other preamble/marker (starts
                //                with a 0x12345678 sentinel, otherwise mostly
                //                zero) — NOT part of the shader. The real,
                //                complete 80-byte hardware SPH (matching
                //                sizeof(Shader::ProgramHeader) exactly) starts
                //                at +48, immediately followed by the actual
                //                Maxwell instructions — i.e. process_blob() can
                //                be handed this slice directly with no
                //                synthesized header at all.
                //
                // All offsets above are absolute from `base` (byte 0 of the
                // BNSH blob, i.e. its own magic) — confirmed via the reference
                // loader's SeekBegin()/TemporarySeek() semantics, which always
                // seek from the start of whatever stream/blob is being read,
                // not relative to any intermediate structure.
                constexpr size_t kByteCodePreambleSize = 48;

                const auto read_u16 = [&](size_t off, u16& out) {
                    if (off + 2 > sz) return false;
                    std::memcpy(&out, data + off, 2); return true;
                };
                const auto read_u32 = [&](size_t off, u32& out) {
                    if (off + 4 > sz) return false;
                    std::memcpy(&out, data + off, 4); return true;
                };
                const auto read_u64 = [&](size_t off, u64& out) {
                    if (off + 8 > sz) return false;
                    std::memcpy(&out, data + off, 8); return true;
                };

                const auto process_stage_offset = [&](size_t base, u64 stage_offset,
                                                       const PreviousStageStoresSnapshot* previous_stage,
                                                       PreviousStageStoresSnapshot* out_stage_snapshot,
                                                       Shader::Backend::Bindings starting_binding,
                                                       Shader::Backend::Bindings* out_end_binding) -> bool {
                    if (stage_offset == 0) return false;
                    const size_t code_hdr = base + static_cast<size_t>(stage_offset);
                    u64 control_code_offset{};
                    u64 byte_code_offset{};
                    u32 byte_code_size{};
                    u32 control_code_size{};
                    if (!read_u64(code_hdr + 0x08, control_code_offset)) return false;
                    if (!read_u64(code_hdr + 0x10, byte_code_offset)) return false;
                    if (!read_u32(code_hdr + 0x18, byte_code_size)) return false;
                    if (!read_u32(code_hdr + 0x1C, control_code_size)) return false;

                    bool succeeded = false;
                    if (byte_code_offset != 0 && byte_code_size > kByteCodePreambleSize) {
                        const size_t blob_start = base + static_cast<size_t>(byte_code_offset) + kByteCodePreambleSize;
                        const size_t blob_size = byte_code_size - kByteCodePreambleSize;
                        if (blob_start + blob_size <= sz) {
                            succeeded = process_blob(std::vector<u8>(data + blob_start, data + blob_start + blob_size),
                                                      /*is_bnsh_derived=*/true, previous_stage, out_stage_snapshot,
                                                      starting_binding, out_end_binding);
                        }
                    }
                    if (succeeded) return true;

                    // Fallback: try ControlCode instead of ByteCode. This mirrors a
                    // workaround documented by the Switch-modding community for
                    // exactly this situation (see e.g. the GBATemp "Dump Vertex and
                    // Fragment Shader code" tutorial, and Switch-Toolbox's own BNSH
                    // exporter, which offers both "Shader0" [ControlCode] and
                    // "Shader1" [ByteCode] as export options because either one can
                    // turn out to be the section that actually contains valid,
                    // decodable code for a given shader — "if [Shader1] gives an
                    // error, try [Shader0]"). Root cause: the real live pipeline's
                    // CFG starts reading at env.StartAddress() + sizeof(SPH), where
                    // StartAddress() is live GPU-register-supplied context this
                    // static scan has no access to — so for some shaders, what we'd
                    // naively pick (ByteCode, offset 0) isn't the section the real
                    // pipeline actually executes from. Trying ControlCode next is a
                    // cheap, community-precedented second guess, not a blind one.
                    if (control_code_offset != 0 && control_code_size > kByteCodePreambleSize) {
                        const size_t blob_start = base + static_cast<size_t>(control_code_offset) + kByteCodePreambleSize;
                        const size_t blob_size = control_code_size - kByteCodePreambleSize;
                        if (blob_start + blob_size <= sz) {
                            succeeded = process_blob(std::vector<u8>(data + blob_start, data + blob_start + blob_size),
                                                      /*is_bnsh_derived=*/true, previous_stage, out_stage_snapshot,
                                                      starting_binding, out_end_binding);
                        }
                    }
                    return succeeded;
                };

                const auto process_bnsh_at = [&](size_t base) {
                    u16 block_offset{};
                    if (!read_u16(base + 0x16, block_offset)) return;
                    const size_t grsc = base + block_offset;
                    u32 grsc_magic{};
                    if (!read_u32(grsc, grsc_magic)) return;
                    if (grsc_magic != 0x63737267u) return; // "grsc"
                    u32 num_variations{};
                    u64 variation_start{};
                    if (!read_u32(grsc + 0x1C, num_variations)) return;
                    if (!read_u64(grsc + 0x20, variation_start)) return;
                    if (num_variations == 0) return;
                    // Sanity bound: reject only what genuinely can't fit — every
                    // variation header is 64 bytes, so more variations than the
                    // remaining buffer could possibly hold is corrupt data, not a
                    // real title. A fixed constant here (this used to be a hardcoded
                    // `> 4096`) doesn't scale: BOTW's uking_mat.product.sbfsha (the
                    // game-wide material shader archive) legitimately has 13,188
                    // variations — verified by sampling variation-table entries
                    // across the full range and confirming every one resolves to an
                    // in-bounds, sane-looking offset — and a 4096 cap silently
                    // dropped the entire file, the single largest source of missed
                    // shaders found during the BOTW/SMO precache investigation.
                    const size_t variation_table_start = base + static_cast<size_t>(variation_start);
                    if (variation_table_start < base || variation_table_start >= sz) return; // overflow/OOB
                    const u64 max_plausible_variations = (sz - variation_table_start) / 64;
                    if (num_variations > max_plausible_variations) return;

                    for (u32 v = 0; v < num_variations; ++v) {
                        const size_t var_hdr = base + static_cast<size_t>(variation_start) + v * 64;
                        u64 binary_offset{};
                        if (!read_u64(var_hdr + 0x10, binary_offset)) continue;
                        if (binary_offset == 0) continue;

                        const size_t prog_hdr = base + static_cast<size_t>(binary_offset);
                        // Scanner runtime-state snapshot: fields 0-4 (Vertex, TessControl,
                        // TessEval, Geometry, Fragment) are a real graphics pipeline
                        // chain within this one shader program, in this order (matching
                        // BnshShaderProgramHeader's own layout) -- processed here in
                        // that same order so each stage's speculative RuntimeInfo can
                        // use the ACTUAL preceding stage's real stores/legacy-stores/
                        // passthrough data (see PreviousStageStoresSnapshot's doc
                        // comment above process_blob) instead of the "no previous
                        // program" sentinel, for every stage that isn't VertexB.
                        // Field 5 (Compute) is a completely separate pipeline type with
                        // no previous-stage concept at all, so it's handled after this
                        // loop, deliberately excluded from the chain in both directions.
                        static constexpr size_t kGraphicsStageOffsetFields[5] = {0x08, 0x10, 0x18, 0x20, 0x28};
                        std::optional<PreviousStageStoresSnapshot> previous_stage_snapshot;
                        Shader::Backend::Bindings binding{};
                        for (const size_t field : kGraphicsStageOffsetFields) {
                            u64 stage_offset{};
                            if (!read_u64(prog_hdr + field, stage_offset)) continue;
                            PreviousStageStoresSnapshot this_stage_snapshot{};
                            Shader::Backend::Bindings end_binding{};
                            const bool succeeded = process_stage_offset(
                                base, stage_offset,
                                previous_stage_snapshot ? &*previous_stage_snapshot : nullptr,
                                &this_stage_snapshot, binding, &end_binding);
                            if (succeeded) {
                                previous_stage_snapshot = this_stage_snapshot;
                                binding = end_binding;
                            }
                            // An absent stage (stage_offset==0, already skipped by the
                            // `continue` above) or one that failed to translate leaves
                            // previous_stage_snapshot exactly as it was -- correctly
                            // chaining through to whichever real stage precedes it, the
                            // same way a real pipeline naturally skips an absent stage
                            // (see MakeRuntimeInfo's own previous_program parameter,
                            // which is computed the same way on the live path).
                        }
                        u64 compute_offset{};
                        if (read_u64(prog_hdr + 0x30, compute_offset) && compute_offset != 0) {
                            process_stage_offset(base, compute_offset, /*previous_stage=*/nullptr,
                                                 /*out_stage_snapshot=*/nullptr,
                                                 Shader::Backend::Bindings{}, nullptr);
                        }
                    }
                };


                // Full-buffer scan, 4-byte-aligned (the format is never sub-word
                // aligned in practice). This runs once per file, on the
                // decompressed buffer where applicable — if profiling later shows
                // this dominates scan time on very large archives, the memcpy+
                // compare per position here is the first thing to optimize (e.g.
                // via std::search / an SSE-friendly scan), but it hasn't been
                // measured yet so this stays simple until there's a reason not to.
                bool any_bnsh = false;
                for (size_t base = 0; base + 4 <= sz; base += 4) {
                    u32 magic4{}; std::memcpy(&magic4, data+base, 4);
                    if (magic4 == BNSH_MAGIC) {
                        any_bnsh = true;
                        process_bnsh_at(base);
                    }
                }

                // Raw-NVN-shader convention: a shader as passed directly to the
                // NVN graphics API (no BNSH/GRSC container at all) is commonly
                // preceded by the literal magic bytes 78 56 34 12, then a fixed
                // 0x30-byte (48-byte) NVN-specific header, then a bare Maxwell
                // ProgramHeader. Confirmed independently by two unrelated public
                // sources (DCNick3/shader-compiler-rs's README; a 2019 GBAtemp
                // hex-editor tutorial for manually dumping Switch shaders) and
                // verified against real game data for Skyward Sword HD (4,192/
                // 4,192 magic occurrences produced a valid shader — 100% hit
                // rate) and structurally matches the same pre-header shape found
                // in Metroid Prime Remastered's SNVN-tagged shaders. Unlike
                // scan_full_for_raw_sph below, this is anchored on a specific,
                // literal magic byte sequence rather than a loose bitfield check
                // alone, so it's safe to apply unconditionally to every title
                // rather than gating it behind a per-format flag — the false-
                // positive risk of stumbling onto this exact 4-byte sequence by
                // chance, immediately followed by 0x30 bytes leading into
                // something that also happens to look like a plausible SPH, is
                // low enough not to need scoping the way HWDE's pairtable-only
                // full scan does. See docs/precache-scanner/FINDINGS.md section 5.
                constexpr u32 kRawNvnShaderMagic = 0x12345678u; // bytes 78 56 34 12 read as LE u32
                constexpr size_t kNvnHeaderSize = 0x30;
                const auto has_valid_sph = [](const Shader::ProgramHeader& sph) noexcept {
                    const u32 type = sph.common0.shader_type.Value();
                    return sph.common0.version.Value() != 0 &&
                           sph.common0.sass_version.Value() != 0 && type >= 1 && type <= 5 &&
                           sph.common0.sph_type.Value() == ((type == 5) ? 2u : 1u);
                };
                bool any_raw_nvn = false;
                if (!any_bnsh) {
                    for (size_t base = 0; base + 4 <= sz; base += 4) {
                        u32 magic4{}; std::memcpy(&magic4, data+base, 4);
                        if (magic4 != kRawNvnShaderMagic) continue;
                        const size_t sph_off = base + kNvnHeaderSize;
                        if (sph_off + sizeof(Shader::ProgramHeader) > sz) continue;
                        Shader::ProgramHeader sph{};
                        std::memcpy(&sph, data + sph_off, sizeof(sph));
                        if (has_valid_sph(sph)) {
                            any_raw_nvn = true;
                            // No BNSH shader-program grouping for a raw-NVN scan
                            // unit either (same reasoning as the pre-existing
                            // offset-0 raw_matched call site above): nullptr for
                            // both new params.
                            process_blob(std::vector<u8>(data + sph_off, data + sz),
                                         /*is_bnsh_derived=*/false,
                                         /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr,
                                         Shader::Backend::Bindings{}, nullptr);
                        }
                    }
                }

                if (any_bnsh) {
                    ++state->bnsh_matched;
                } else if (any_raw_nvn) {
                    ++state->raw_matched;
                } else if (scan_unit.scan_full_for_raw_sph) {
                    // Pairtable-derived blocks (Hyrule Warriors Definitive Edition,
                    // and per the investigation this is based on, plausibly sibling
                    // Omega Force Switch titles) hold a variable-length reflection/
                    // metadata header before the real, bare Maxwell ProgramHeader —
                    // unlike every other title supported so far, that header is
                    // never at offset 0, so every 4-byte-aligned offset needs
                    // checking instead of just one. Deliberately scoped to only
                    // these blocks rather than applying to every file's raw-match
                    // check unconditionally: this is a materially more expensive
                    // per-offset scan than a single check, and applying it
                    // everywhere would raise both scan time and false-positive
                    // exposure for every title that doesn't need it. See
                    // docs/precache-scanner/FINDINGS.md section 4.
                    bool any_raw_matched = false;
                    for (size_t off = 0; off + sizeof(Shader::ProgramHeader) <= sz; off += 4) {
                        Shader::ProgramHeader sph{};
                        std::memcpy(&sph, data + off, sizeof(sph));
                        if (has_valid_sph(sph)) {
                            any_raw_matched = true;
                            // Same reasoning as the raw-NVN call site above: no
                            // sibling-stage data for a pairtable-derived scan unit.
                            process_blob(std::vector<u8>(data + off, data + sz),
                                         /*is_bnsh_derived=*/false,
                                         /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr,
                                         Shader::Backend::Bindings{}, nullptr);
                        }
                    }
                    if (any_raw_matched) {
                        ++state->raw_matched;
                    } else {
                        ++state->unrecognized;
                        log_unrecognized_sample();
                    }
                } else if (sz>=sizeof(Shader::ProgramHeader)) {
                    Shader::ProgramHeader sph{}; std::memcpy(&sph,data,sizeof(sph));
                    if (has_valid_sph(sph)) {
                        ++state->raw_matched;
                        // No BNSH shader-program grouping for a standalone raw-matched
                        // file, so no real sibling-stage data exists here -- nullptr for
                        // both fall back to the prior sentinel
                        // behavior for this path, unchanged.
                        process_blob(std::vector<u8>(data, data+sz), /*is_bnsh_derived=*/false,
                                     /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr,
                                     Shader::Backend::Bindings{}, nullptr);
                    } else {
                        ++state->unrecognized;
                        log_unrecognized_sample();
                    }
                } else {
                    ++state->unrecognized;
                    log_unrecognized_sample();
                }
                } // for (scan_unit : scan_units)
            });
        }
        workers.WaitForRequests();
        std::vector<VideoCommon::PrecacheCfgArtifact> cfg_artifact_snapshot;
        {
            std::lock_guard lock{state->cfg_artifact_mutex};
            cfg_artifact_snapshot = state->cfg_artifacts;
        }
        const bool cfg_artifacts_saved = !cfg_artifact_snapshot.empty() &&
                                         VideoCommon::SavePrecacheCfgArtifacts(
                                             cache_dir / fmt::format(
                                                 "precache_cfg_artifacts_{:016x}.bin",
                                                 compiler_target_key),
                                             cfg_artifact_snapshot);
        std::vector<VideoCommon::PrecacheFrontendArtifactRecord> frontend_artifact_snapshot;
        {
            std::lock_guard lock{state->frontend_artifact_mutex};
            frontend_artifact_snapshot = state->frontend_artifacts;
        }
        const bool frontend_artifacts_saved = !frontend_artifact_snapshot.empty() &&
                                              VideoCommon::SavePrecacheFrontendArtifacts(
                                                  cache_dir / fmt::format(
                                                      "precache_frontend_artifacts_{:016x}.bin",
                                                      compiler_target_key),
                                                  frontend_artifact_snapshot);
        cache.Save(spirv_path);
        LOG_INFO(Render_Vulkan,
                 "PreCacheShaders: done. files={} processed={} zstd_decompressed={} "
                 "zstd_failed={} bnsh_matched={} raw_matched={} unrecognized={} "
                 "shaders_found={} translated={} final_entries={} alignment_translated_entries={} "
                 "unknown_deps(cbuf/texture/viewport/compute/hle_macro/interface/scheduling_alignment)={}/{}/{}/{}/{}/{}/{} "
                 "final_contract_rejections(cbuf/texture/viewport/compute/hle_macro/interface/scheduling_alignment/identity)={}/{}/{}/{}/{}/{}/{}/{} "
                 "template_boundary(eligible/reject_cbuf/reject_texture/reject_viewport/reject_compute/reject_hle_macro/reject_interface/reject_scheduling_alignment)={}/{}/{}/{}/{}/{}/{}/{} "
                 "frontend_manifest(cbuf/resource/viewport/compute/hle/interface)={}/{}/{}/{}/{}/{} "
                 "cfg_artifacts(candidates/stored/saved/cbuf/indirect/capacity)={}/{}/{}/{}/{}/{} "
                 "frontend_artifacts(candidates/stored/saved/info/pseudo/value/graph/capacity)={}/{}/{}/{}/{}/{}/{}/{} "
                 "frontend_artifact_stages(VB/TC/TE/G/F/C/VA)={}/{}/{}/{}/{}/{}/{} "
                 "cfg_artifact_stages(VB/TC/TE/G/F/C/VA)={}/{}/{}/{}/{}/{}/{} "
                 "failed={} cache_size={} texture_key_mode(logical/raw_fallback)={}/{} "
                 "worker_cpu_ms(cfg/template/finalize/emit)={}/{}/{}/{}",
                 state->files_total.load(), state->files_processed.load(),
                 state->zstd_decompressed.load(), state->zstd_failed.load(),
                 state->bnsh_matched.load(), state->raw_matched.load(),
                 state->unrecognized.load(), state->shaders_found.load(),
                 state->shaders_translated.load(), state->final_entries.load(),
                 state->alignment_translated_entries.load(), state->unknown_cbuf_candidates.load(),
                 state->unknown_texture_candidates.load(), state->unknown_viewport_candidates.load(),
                 state->unknown_compute_candidates.load(), state->unknown_hle_macro_candidates.load(),
                 state->unknown_interface_candidates.load(),
                 state->unknown_scheduling_alignment_candidates.load(),
                 state->final_cbuf_contract_rejections.load(),
                 state->final_texture_contract_rejections.load(),
                 state->final_viewport_contract_rejections.load(),
                 state->final_compute_contract_rejections.load(),
                 state->final_hle_macro_contract_rejections.load(),
                 state->final_interface_contract_rejections.load(),
                 state->final_scheduling_alignment_contract_rejections.load(),
                 state->final_identity_contract_rejections.load(),
                 state->template_boundary_eligible.load(),
                 state->template_boundary_cbuf_rejections.load(),
                 state->template_boundary_texture_rejections.load(),
                 state->template_boundary_viewport_rejections.load(),
                 state->template_boundary_compute_rejections.load(),
                 state->template_boundary_hle_macro_rejections.load(),
                 state->template_boundary_interface_rejections.load(),
                 state->template_boundary_scheduling_alignment_rejections.load(),
                 state->frontend_manifest_cbuf.load(), state->frontend_manifest_resource.load(),
                 state->frontend_manifest_viewport.load(),
                 state->frontend_manifest_compute.load(), state->frontend_manifest_hle.load(),
                 state->frontend_manifest_interface.load(),
                 state->cfg_artifact_candidates.load(), state->cfg_artifact_stored.load(),
                 cfg_artifacts_saved ? 1 : 0, state->cfg_artifact_cbuf_rejections.load(),
                 state->cfg_artifact_branches_rejections.load(),
                 state->cfg_artifact_capacity_rejections.load(),
                 state->frontend_artifact_candidates.load(),
                 state->frontend_artifact_stored.load(), frontend_artifacts_saved ? 1 : 0,
                 state->frontend_artifact_info_rejections.load(),
                 state->frontend_artifact_pseudo_rejections.load(),
                 state->frontend_artifact_value_rejections.load(),
                 state->frontend_artifact_graph_rejections.load(),
                 state->frontend_artifact_capacity_rejections.load(),
                 state->frontend_artifact_stages[0], state->frontend_artifact_stages[1],
                 state->frontend_artifact_stages[2], state->frontend_artifact_stages[3],
                 state->frontend_artifact_stages[4], state->frontend_artifact_stages[5],
                 state->frontend_artifact_stages[6],
                 state->cfg_artifact_stages[0], state->cfg_artifact_stages[1],
                 state->cfg_artifact_stages[2], state->cfg_artifact_stages[3],
                 state->cfg_artifact_stages[4], state->cfg_artifact_stages[5],
                 state->cfg_artifact_stages[6],
                 state->shaders_failed.load(),
                 cache.Size(), cache.LogicalTextureKeyCount(), cache.RawTextureKeyCount(),
                 state->cfg_us.load() / 1000, state->template_us.load() / 1000,
                 state->finalize_us.load() / 1000, state->emit_us.load() / 1000);
    };

    auto future = QtConcurrent::run(std::move(worker));

    QTimer poll;
    connect(&poll, &QTimer::timeout, [&]() {
        if (progress.wasCanceled()) state->cancelled = true;
        const int total     = state->files_total.load();
        const int processed = state->files_processed.load();
        const int found     = state->shaders_found.load();
        const int translated= state->shaders_translated.load();
        if (total==0) {
            progress.setMaximum(0);
            // The normal path already captured a resolved RomFS directory
            // during the short boot. Until the background worker has finished
            // indexing its files, total is zero; this is not a second mount.
            progress.setLabelText(tr("Indexing resolved RomFS files..."));
        } else if (processed<total) {
            progress.setMaximum(total); progress.setValue(processed);
            progress.setLabelText(tr("Scanning files... %1 / %2  (%3 shaders found)")
                .arg(processed).arg(total).arg(found));
        } else {
            progress.setMaximum(found>0?found:1); progress.setValue(translated);
            progress.setLabelText(tr("Translating shaders... %1 / %2")
                .arg(translated).arg(found));
        }
        if (future.isFinished()) {
            poll.stop();
            final_result = {state->shaders_translated.load(),
                            state->shaders_failed.load(),
                            state->error_message,
                            state->cancelled.load()};
            progress.accept();
        }
    });
    poll.start(100);
    progress.exec();
    future.waitForFinished();

    if (!final_result.error.empty()) {
        QMessageBox::critical(this, tr("Pre-cache Shaders"),
            tr("Shader scan failed:\n%1").arg(QString::fromStdString(final_result.error)));
    } else if (final_result.cancelled) {
        QMessageBox::information(this, tr("Pre-cache Shaders"),
            tr("Pre-cache cancelled. %1 shaders translated so far have been saved.")
                .arg(final_result.translated));
    } else {
        QMessageBox::information(this, tr("Pre-cache Shaders"),
            tr("Pre-cache complete!\n\n"
               "New shaders translated:    %1\n"
               "Failed (runtime fallback): %2\n\n"
               "Shaders already in cache are skipped automatically.\n"
               "SPIR-V cache saved — stutter will be reduced on first play.")
                .arg(final_result.translated)
                .arg(final_result.failed));
    }
}
