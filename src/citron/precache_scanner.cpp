// SPDX-FileCopyrightText: 2025 citron Emulator Project
// SPDX-FileCopyrightText: 2026 citron-neo Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GMainWindow::OnGameListPreCacheShaders — the boot-time shader pre-cache scanner.
// Extracted from main.cpp, where it grew to ~1,400 lines across several rounds of
// work: the original BNSH/GRSC/SPH scanner, Phase 3's real previous-stage-stores
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

#include <QMessageBox>
#include <QProgressDialog>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>

#include <fmt/format.h>

#include "citron/main.h"

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/thread_worker.h"
#include "common/crilayla_compression.h"
#include "common/xbc1_compression.h"
#include "common/yaz0_compression.h"
#include "common/zlib_compression.h"
#include "common/zstd_compression.h"

#include "core/loader/loader.h"
#include "core/loader/nca.h"
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

#include "video_core/speculative_shader_environment.h"
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

// Phase 3 guess refinement. Snapshot of exactly the fields MakeRuntimeInfo()
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

// One already-unwrapped region of bytes ready for the BNSH/GRSC/SPH scan below —
// either a whole (post zstd/Yaz0) file, one named entry out of a SARC archive, or
// one block out of a pairtable-wrapped file (Hyrule Warriors Definitive Edition
// and, per the investigation this is based on, plausibly sibling Omega Force
// titles). Built as a list rather than recursing so the existing scan/classify
// logic runs unmodified, once per unit -- a plain file still produces exactly
// one unit, so this is a no-op for every title that doesn't need container
// unwrapping at this stage (TotK included). See
// docs/precache-scanner/FINDINGS.md for the full investigation behind SARC and
// pairtable support specifically.
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

    const auto shader_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::ShaderDir);
    const auto cache_dir  = shader_dir / fmt::format("{:016x}", program_id);
    if (!Common::FS::CreateDirs(cache_dir)) return;
    const auto spirv_path = cache_dir / "spirv_cache.bin";

    const int kMaxSamples = 8;
    const int kMaxDiagBlobs = 3000;

    struct ScanState {
        std::atomic<int>  files_total{0};
        std::atomic<int>  files_processed{0};
        std::atomic<int>  shaders_found{0};
        std::atomic<int>  shaders_translated{0};
        std::atomic<int>  shaders_failed{0};
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
    };
    auto state = std::make_shared<ScanState>();

    struct FinalResult { int translated, failed; std::string error; bool cancelled; };
    FinalResult final_result{};

    QProgressDialog progress(tr("Opening game file..."),
                             tr("Cancel"), 0, 0, this);
    progress.setWindowTitle(tr("Pre-cache Shaders"));
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setAutoClose(false);
    progress.setAutoReset(false);

    auto local_vfs = std::make_shared<FileSys::RealVfsFilesystem>();
    const auto game_file = local_vfs->OpenFile(game_path, FileSys::OpenMode::Read);

    auto worker = [state, game_path, spirv_path, game_file]() {
        // Mount RomFS
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
        const auto romfs = FileSys::ExtractRomFS(romfs_raw);
        if (!romfs) {
            state->error_message = "Failed to extract RomFS.";
            LOG_ERROR(Render_Vulkan, "PreCacheShaders: {}", state->error_message);
            return;
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
        if (const auto zsdic_file = romfs->GetFileRelative("Pack/ZsDic.pack.zs")) {
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
        walk(romfs);

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

        Shader::Profile profile{};
        profile.supported_spirv=0x00010300; profile.unified_descriptor_binding=true;
        profile.support_descriptor_aliasing=true;
        profile.support_int8=profile.support_int16=profile.support_int64=true;
        profile.support_float_controls=true; profile.support_vote=true;
        profile.support_typeless_image_loads=true;
        profile.support_demote_to_helper_invocation=true;
        profile.min_ssbo_alignment=16; profile.max_user_clip_distances=8;
        Shader::HostTranslateInfo host_info{};
        host_info.support_float64=host_info.support_float16=host_info.support_int64=true;
        host_info.support_snorm_render_buffer=true;
        host_info.support_viewport_index_layer=true;
        host_info.min_ssbo_alignment=16;
        // Always disable conditional barrier support: shaders cached here may be loaded
        // on Intel Windows drivers where barriers inside conditional control flow are
        // illegal.  Stripping barriers from conditional CF is always spec-correct, so
        // this is safe on all drivers and avoids generating SPIR-V that would be
        // rejected or miscompiled on the target hardware.
        host_info.support_conditional_barrier=false;

        std::mutex seen_mutex;
        // Checked/inserted while holding seen_mutex, on every shader blob
        // candidate found across the scan (potentially many thousands between
        // BNSH sub-blocks and raw-matched files) — unordered_dense's better
        // cache locality reduces time spent inside the lock, which matters
        // more here than usual since it's contended across every worker
        // thread, not just a single-thread hot path.
        ankerl::unordered_dense::set<u64> seen_hashes;

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
                }
                if (sz < 4) return;

                // Phase 3 guess refinement. Snapshot of exactly the fields
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
                                              PreviousStageStoresSnapshot* out_stage_snapshot) -> bool {
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
                    if (version == 0u)              { if (diag_slot >= 0) LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: rejected — version == 0", diag_slot); return false; }
                    if (shader_type < 1u || shader_type > 5u) { if (diag_slot >= 0) LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: rejected — shader_type out of range", diag_slot); return false; }
                    const u32 expected_sph_type = (shader_type == 5u) ? 2u : 1u;
                    if (sph_type != expected_sph_type) { if (diag_slot >= 0) LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: rejected — sph_type={} (expected {} for shader_type={})", diag_slot, sph_type, expected_sph_type, shader_type); return false; }
                    if (sass_ver == 0u)             { if (diag_slot >= 0) LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: rejected — sass_ver == 0", diag_slot); return false; }

                    // Require at least one instruction beyond the header,
                    // aligned to 8 bytes (Maxwell instruction size).
                    const size_t payload = blob.size() - sizeof(Shader::ProgramHeader);
                    if (payload < 8 || payload % 8 != 0) {
                        if (diag_slot >= 0) {
                            LOG_INFO(Render_Vulkan,
                                     "PreCacheShaders diag[{}]: rejected — payload={} not 8-aligned "
                                     "or empty", diag_slot, payload);
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

                    // Quick same-session dedup using a blob fingerprint.
                    // This prevents re-translating identical blobs found in multiple
                    // RomFS files within a single scan. It is NOT the authoritative
                    // shader hash — that comes from env.CalculateHash() below.
                    const u64 blob_fingerprint = Common::CityHash64(
                        reinterpret_cast<const char*>(blob.data()), blob.size());
                    { std::lock_guard g{seen_mutex};
                      if (!seen_hashes.insert(blob_fingerprint).second) return true; }

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
                    // at start_address, which IS the SPH's own address there). This is
                    // what code_lowest=0 in the scanner's SpeculativeShaderEnvironment
                    // constructor now assumes; see that constructor's doc comment for
                    // why this alignment is what makes CalculateHash() actually agree
                    // with GenericEnvironment::Analyze() on the same shader.
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

                    // Attempts a full translate — CFG, TranslateProgram, and SPIR-V
                    // emission — starting from a given byte offset WITHIN the
                    // payload (0 = the naive/default assumption: real code starts
                    // immediately after the SPH). This is a much stronger success
                    // signal than just checking whether a few instructions decode
                    // without throwing: CFG construction validates branch targets
                    // resolve to in-bounds instruction boundaries, block structure
                    // is self-consistent, etc. — a wrong alignment is very unlikely
                    // to satisfy all of that by chance. code is passed by value
                    // (copied, not moved) so it can be reused across multiple
                    // attempts at different offsets.
                    const auto try_translate_at = [&](u32 entry_offset_in_payload) -> bool {
                        try {
                            VideoCommon::SpeculativeShaderEnvironment env{code, stage, lm, bsph};
                            Shader::ObjectPool<Shader::Maxwell::Flow::Block> fp(16);
                            Shader::ObjectPool<Shader::IR::Inst> ip(8192);
                            Shader::ObjectPool<Shader::IR::Block> bp(32);
                            const u32 start_address =
                                static_cast<u32>(sizeof(Shader::ProgramHeader)) + entry_offset_in_payload;
                            Shader::Maxwell::Flow::CFG cfg(env, fp, start_address, false);
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: CFG construction OK at entry +{}",
                                         diag_slot, entry_offset_in_payload);
                            }

                            // Compute the authoritative hash AFTER CFG determines shader bounds.
                            // This must match GenericEnvironment::CalculateHash() used by the live path.
                            const u64 unique_hash = env.CalculateHash();
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: unique_hash={:016x} already_cached={}",
                                         diag_slot, unique_hash, cache.ContainsByUniqueHash(unique_hash));
                            }
                            if (cache.ContainsByUniqueHash(unique_hash)) return true;
                            ++state->shaders_found;

                            auto prog = Shader::Maxwell::TranslateProgram(ip,bp,env,cfg,host_info);
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: TranslateProgram OK at entry +{}",
                                         diag_slot, entry_offset_in_payload);
                            }
                            Shader::Backend::Bindings binding{};
                            Shader::RuntimeInfo rt{};
                            // Phase 3 guess refinement: for every stage except VertexB,
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
                            // Phase 5 free wins (runtime_info.h) -- deliberate defaults for
                            // the fields this speculative path has no real per-draw signal
                            // for.
                            rt.ApplySpeculativeDefaults(stage, prog.info);
                            Shader::Maxwell::ConvertLegacyToGeneric(prog, rt);
                            auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile,rt,prog,binding);
                            const u64 texture_key = VideoCommon::ComputeTextureKey(env.CapturedTextureTypes(), env.CapturedTexturePixelFormats());
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
                            u64 runtime_key = rt.SpirvRelevantHash(stage);
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
                            const u64 diag_binding_key =
                                VideoCommon::ComputeBindingKey(Shader::Backend::Bindings{});
                            runtime_key = VideoCommon::FoldBindingKey(runtime_key, diag_binding_key);
                            cache.InsertSpeculative(unique_hash, runtime_key, texture_key, std::move(spirv),
                                                    diag_base_runtime_hash, diag_binding_key);
                            ++state->shaders_translated;
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: fully translated OK at entry +{}",
                                         diag_slot, entry_offset_in_payload);
                            }
                            // Written only here, on confirmed full success (translate +
                            // emit + insert all completed) -- not right after
                            // TranslateProgram above, so a later failure/exception in
                            // this same attempt can never leave the caller thinking this
                            // stage succeeded when process_blob is about to return false.
                            if (out_stage_snapshot) {
                                *out_stage_snapshot = PreviousStageStoresSnapshot{
                                    prog.info.stores, prog.info.legacy_stores_mapping,
                                    prog.info.passthrough, prog.is_geometry_passthrough};
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
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: entry +{} threw std::exception: {}",
                                         diag_slot, entry_offset_in_payload, e.what());
                            }
                            return false;
                        } catch (...) {
                            if (diag_slot >= 0) {
                                LOG_INFO(Render_Vulkan, "PreCacheShaders diag[{}]: entry +{} threw unknown exception",
                                         diag_slot, entry_offset_in_payload);
                            }
                            return false;
                        }
                    };

                    if (try_translate_at(0)) {
                        return true;
                    }

                    // On by default as of this build — was opt-in behind
                    // CITRON_PRECACHE_BRUTEFORCE_ENTRY=1 while this was still being
                    // validated. It's still expensive (each candidate re-runs the
                    // whole translate pipeline, not a cheap decode check), so a full
                    // ROM scan takes meaningfully longer than the naive-offset-only
                    // pass, but that's the intended default now rather than
                    // something to remember to opt into per session.
                    // Set CITRON_PRECACHE_BRUTEFORCE_ENTRY=0 to disable it (e.g. to
                    // fall back to the old, faster, naive-offset-only behavior).
                    // Window and step are in bytes; CITRON_PRECACHE_FILTER remains
                    // useful for restricting a scan to a small, known set of files.
                    static const bool bruteforce_entry = [] {
                        const char* v = std::getenv("CITRON_PRECACHE_BRUTEFORCE_ENTRY");
                        return !(v && *v && std::string(v) == "0");
                    }();
                    if (bruteforce_entry) {
                        // Default window bumped from 512 to 8192 bytes: a scan with
                        // the old 4096-byte default logged failed=9389, of which 507
                        // were specifically "no working entry within +4096 bytes" —
                        // i.e. window-limited, not a translation failure. 8192 gives
                        // that tail more room without needing CITRON_PRECACHE_
                        // BRUTEFORCE_WINDOW set by hand. Don'''t expect this to move
                        // `failed` by much on its own, though — the other ~8900
                        // failures in that run were for unrelated reasons (unsupported
                        // instructions, genuine translation errors, etc.) a bigger
                        // window can'''t fix. Still overridable via the env var, larger
                        // or smaller, same as before.
                        u32 window_bytes = 8192;
                        if (const char* w = std::getenv("CITRON_PRECACHE_BRUTEFORCE_WINDOW")) {
                            const int parsed = std::atoi(w);
                            if (parsed > 0) window_bytes = static_cast<u32>(parsed);
                        }
                        const u32 limit = std::min<u32>(window_bytes, static_cast<u32>(payload));

                        // Fast pass: across two independent stages (vertex and
                        // fragment) and 55 real successes found scanning every
                        // 8-byte position, every single one landed at
                        // payload_offset % 32 == 8, with zero exceptions. That's
                        // not coincidence at that sample size — it's a real
                        // structural fact (almost certainly the entry point
                        // sitting right after a variable-length, 32-byte-granular
                        // embedded-constants region — consistent with e.g. 4x4
                        // matrices or similar GPU-aligned data preceding the real
                        // code). Try this residue class first: same recall so
                        // far, 4x fewer candidates.
                        for (u32 off = 8; off < limit; off += 32) {
                            if (try_translate_at(off)) {
                                if (diag_slot >= 0) {
                                    LOG_INFO(Render_Vulkan,
                                             "PreCacheShaders diag[{}]: BRUTEFORCE (fast pass, %32==8) found "
                                             "working entry at payload offset +{} (naive offset was +0)",
                                             diag_slot, off);
                                }
                                return true;
                            }
                        }
                        // Fallback: exhaustive search of the remaining phases, in
                        // case some shader's true entry doesn't fit the pattern
                        // above. Only reached when the fast pass above found
                        // nothing, so this doesn't cost anything extra for the
                        // (so far) common case.
                        for (u32 off = 0; off < limit; off += 8) {
                            if (off % 32 == 8) continue; // already tried above
                            if (try_translate_at(off)) {
                                if (diag_slot >= 0) {
                                    LOG_INFO(Render_Vulkan,
                                             "PreCacheShaders diag[{}]: BRUTEFORCE (fallback pass) found "
                                             "working entry at payload offset +{} (naive offset was +0, "
                                             "did NOT fit the %32==8 pattern)",
                                             diag_slot, off);
                                }
                                return true;
                            }
                        }
                        if (diag_slot >= 0) {
                            LOG_INFO(Render_Vulkan,
                                     "PreCacheShaders diag[{}]: BRUTEFORCE found no working entry within "
                                     "+{} bytes", diag_slot, limit);
                        }
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
                                                       PreviousStageStoresSnapshot* out_stage_snapshot) -> bool {
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
                                                      /*is_bnsh_derived=*/true, previous_stage, out_stage_snapshot);
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
                                                      /*is_bnsh_derived=*/true, previous_stage, out_stage_snapshot);
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
                        // Phase 3 guess refinement: fields 0-4 (Vertex, TessControl,
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
                        for (const size_t field : kGraphicsStageOffsetFields) {
                            u64 stage_offset{};
                            if (!read_u64(prog_hdr + field, stage_offset)) continue;
                            PreviousStageStoresSnapshot this_stage_snapshot{};
                            const bool succeeded = process_stage_offset(
                                base, stage_offset,
                                previous_stage_snapshot ? &*previous_stage_snapshot : nullptr,
                                &this_stage_snapshot);
                            if (succeeded) {
                                previous_stage_snapshot = this_stage_snapshot;
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
                                                 /*out_stage_snapshot=*/nullptr);
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
                bool any_raw_nvn = false;
                if (!any_bnsh) {
                    for (size_t base = 0; base + 4 <= sz; base += 4) {
                        u32 magic4{}; std::memcpy(&magic4, data+base, 4);
                        if (magic4 != kRawNvnShaderMagic) continue;
                        const size_t sph_off = base + kNvnHeaderSize;
                        if (sph_off + sizeof(Shader::ProgramHeader) > sz) continue;
                        Shader::ProgramHeader sph{};
                        std::memcpy(&sph, data + sph_off, sizeof(sph));
                        if (sph.common0.shader_type.Value() >= 1 && sph.common0.shader_type.Value() <= 5) {
                            any_raw_nvn = true;
                            // No BNSH shader-program grouping for a raw-NVN scan
                            // unit either (same reasoning as the pre-existing
                            // offset-0 raw_matched call site above): nullptr for
                            // both new params.
                            process_blob(std::vector<u8>(data + sph_off, data + sz),
                                         /*is_bnsh_derived=*/false,
                                         /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr);
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
                        if (sph.common0.shader_type.Value() >= 1 && sph.common0.shader_type.Value() <= 5) {
                            any_raw_matched = true;
                            // Same reasoning as the raw-NVN call site above: no
                            // sibling-stage data for a pairtable-derived scan unit.
                            process_blob(std::vector<u8>(data + off, data + sz),
                                         /*is_bnsh_derived=*/false,
                                         /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr);
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
                    if (sph.common0.shader_type.Value()>=1 && sph.common0.shader_type.Value()<=5) {
                        ++state->raw_matched;
                        // No BNSH shader-program grouping for a standalone raw-matched
                        // file, so no real sibling-stage data exists here -- nullptr for
                        // both falls back to exactly the previous (pre-Phase-3) sentinel
                        // behavior for this path, unchanged.
                        process_blob(std::vector<u8>(data, data+sz), /*is_bnsh_derived=*/false,
                                     /*previous_stage=*/nullptr, /*out_stage_snapshot=*/nullptr);
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
        cache.Save(spirv_path);
        LOG_INFO(Render_Vulkan,
                 "PreCacheShaders: done. files={} processed={} zstd_decompressed={} "
                 "zstd_failed={} bnsh_matched={} raw_matched={} unrecognized={} "
                 "shaders_found={} translated={} failed={} cache_size={}",
                 state->files_total.load(), state->files_processed.load(),
                 state->zstd_decompressed.load(), state->zstd_failed.load(),
                 state->bnsh_matched.load(), state->raw_matched.load(),
                 state->unrecognized.load(), state->shaders_found.load(),
                 state->shaders_translated.load(), state->shaders_failed.load(),
                 cache.Size());
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
            progress.setLabelText(tr("Mounting RomFS..."));
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
