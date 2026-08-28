// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <system_error>

#include "common/assert.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/fs_filesystem.h"
#include "core/file_sys/nca_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/system_archive/system_archive.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/frontend/applets/web_browser.h"
#include "core/hle/result.h"
#include "core/hle/service/am/am.h"
#include "core/hle/service/am/frontend/applet_web_browser.h"
#include "core/hle/service/am/service/storage.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/ns/platform_service_manager.h"
#include "core/loader/loader.h"

// [UNITY-FIX] winbase.h A/W macros shadow C++ method names.
#undef DeleteFile
#undef CreateFile
#undef CopyFile
#undef MoveFile
#undef MoveFileEx
#undef CreateDirectory
#undef RemoveDirectory

namespace Service::AM::Frontend {

namespace {

template <typename T>
void ParseRawValue(T& value, const std::vector<u8>& data) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "It's undefined behavior to use memcpy with non-trivially copyable objects");
    std::memcpy(&value, data.data(), data.size());
}

template <typename T>
T ParseRawValue(const std::vector<u8>& data) {
    T value;
    ParseRawValue(value, data);
    return value;
}

std::string ParseStringValue(const std::vector<u8>& data) {
    return Common::StringFromFixedZeroTerminatedBuffer(reinterpret_cast<const char*>(data.data()),
                                                       data.size());
}

std::string GetMainURL(const std::string& url) {
    const auto index = url.find('?');

    if (index == std::string::npos) {
        return url;
    }

    return url.substr(0, index);
}

std::string ResolveURL(const std::string& url) {
    const auto index = url.find_first_of('%');

    if (index == std::string::npos) {
        return url;
    }

    return url.substr(0, index) + "lp1" + url.substr(index + 1);
}

WebArgInputTLVMap ReadWebArgs(const std::vector<u8>& web_arg, WebArgHeader& web_arg_header) {
    std::memcpy(&web_arg_header, web_arg.data(), sizeof(WebArgHeader));

    if (web_arg.size() == sizeof(WebArgHeader)) {
        return {};
    }

    WebArgInputTLVMap input_tlv_map;

    u64 current_offset = sizeof(WebArgHeader);

    for (std::size_t i = 0; i < web_arg_header.total_tlv_entries; ++i) {
        if (web_arg.size() < current_offset + sizeof(WebArgInputTLV)) {
            return input_tlv_map;
        }

        WebArgInputTLV input_tlv;
        std::memcpy(&input_tlv, web_arg.data() + current_offset, sizeof(WebArgInputTLV));

        current_offset += sizeof(WebArgInputTLV);

        if (web_arg.size() < current_offset + input_tlv.arg_data_size) {
            return input_tlv_map;
        }

        std::vector<u8> data(input_tlv.arg_data_size);
        std::memcpy(data.data(), web_arg.data() + current_offset, input_tlv.arg_data_size);

        current_offset += input_tlv.arg_data_size;

        input_tlv_map.insert_or_assign(input_tlv.input_tlv_type, std::move(data));
    }

    return input_tlv_map;
}

FileSys::VirtualFile GetOfflineRomFS(Core::System& system, u64 title_id,
                                     FileSys::ContentRecordType nca_type) {
    if (nca_type == FileSys::ContentRecordType::Data) {
        const auto nca =
            system.GetFileSystemController().GetSystemNANDContents()->GetEntry(title_id, nca_type);

        if (nca == nullptr) {
            LOG_ERROR(Service_AM,
                      "NCA of type={} with title_id={:016X} is not found in the System NAND!",
                      nca_type, title_id);
            return FileSys::SystemArchive::SynthesizeSystemArchive(title_id);
        }

        return nca->GetRomFS();
    } else {
        const auto nca = system.GetContentProvider().GetEntry(title_id, nca_type);

        if (nca == nullptr) {
            if (nca_type == FileSys::ContentRecordType::HtmlDocument) {
                LOG_WARNING(Service_AM, "Falling back to AppLoader to get the RomFS.");
                FileSys::VirtualFile romfs;
                system.GetAppLoader().ReadManualRomFS(romfs);
                if (romfs != nullptr) {
                    return romfs;
                }
            }

            LOG_ERROR(Service_AM,
                      "NCA of type={} with title_id={:016X} is not found in the ContentProvider!",
                      nca_type, title_id);
            return nullptr;
        }

        const FileSys::PatchManager pm{title_id, system.GetFileSystemController(),
                                       system.GetContentProvider()};

        return pm.PatchRomFS(nca.get(), nca->GetRomFS(), nca_type);
    }
}

void ExtractSharedFonts(Core::System& system) {
    static constexpr std::array<const char*, 7> DECRYPTED_SHARED_FONTS{
        "FontStandard.ttf",
        "FontChineseSimplified.ttf",
        "FontExtendedChineseSimplified.ttf",
        "FontChineseTraditional.ttf",
        "FontKorean.ttf",
        "FontNintendoExtended.ttf",
        "FontNintendoExtended2.ttf",
    };

    const auto fonts_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "fonts";

    for (std::size_t i = 0; i < NS::SHARED_FONTS.size(); ++i) {
        const auto font_file_path = fonts_dir / DECRYPTED_SHARED_FONTS[i];

        if (Common::FS::Exists(font_file_path)) {
            continue;
        }

        const auto font = NS::SHARED_FONTS[i];
        const auto font_title_id = static_cast<u64>(font.first);

        const auto nca = system.GetFileSystemController().GetSystemNANDContents()->GetEntry(
            font_title_id, FileSys::ContentRecordType::Data);

        FileSys::VirtualFile romfs;

        if (!nca) {
            romfs = FileSys::SystemArchive::SynthesizeSystemArchive(font_title_id);
        } else {
            romfs = nca->GetRomFS();
        }

        if (!romfs) {
            LOG_ERROR(Service_AM, "SharedFont RomFS with title_id={:016X} cannot be extracted!",
                      font_title_id);
            continue;
        }

        const auto extracted_romfs = FileSys::ExtractRomFS(romfs);

        if (!extracted_romfs) {
            LOG_ERROR(Service_AM, "SharedFont RomFS with title_id={:016X} failed to extract!",
                      font_title_id);
            continue;
        }

        const auto font_file = extracted_romfs->GetFile(font.second);

        if (!font_file) {
            LOG_ERROR(Service_AM, "SharedFont RomFS with title_id={:016X} has no font file \"{}\"!",
                      font_title_id, font.second);
            continue;
        }

        std::vector<u32> font_data_u32(font_file->GetSize() / sizeof(u32));
        font_file->ReadBytes<u32>(font_data_u32.data(), font_file->GetSize());

        std::transform(font_data_u32.begin(), font_data_u32.end(), font_data_u32.begin(),
                       Common::swap32);

        std::vector<u8> decrypted_data(font_file->GetSize() - 8);

        NS::DecryptSharedFontToTTF(font_data_u32, decrypted_data);

        FileSys::VirtualFile decrypted_font = std::make_shared<FileSys::VectorVfsFile>(
            std::move(decrypted_data), DECRYPTED_SHARED_FONTS[i]);

        const auto temp_dir = system.GetFilesystem()->CreateDirectory(
            Common::FS::PathToUTF8String(fonts_dir), FileSys::OpenMode::ReadWrite);

        const auto out_file = temp_dir->CreateFile(DECRYPTED_SHARED_FONTS[i]);

        FileSys::VfsRawCopy(decrypted_font, out_file);
    }
}

} // namespace

WebBrowser::WebBrowser(Core::System& system_, std::shared_ptr<Applet> applet_,
                       LibraryAppletMode applet_mode_,
                       const Core::Frontend::WebBrowserApplet& frontend_)
    : FrontendApplet{system_, applet_, applet_mode_}, frontend(frontend_) {}

WebBrowser::~WebBrowser() = default;

void WebBrowser::Initialize() {
    FrontendApplet::Initialize();

    LOG_INFO(Service_AM, "Initializing Web Browser Applet.");

    LOG_DEBUG(Service_AM,
              "Initializing Applet with common_args: arg_version={}, lib_version={}, "
              "play_startup_sound={}, size={}, system_tick={}, theme_color={}",
              common_args.arguments_version, common_args.library_version,
              common_args.play_startup_sound, common_args.size, common_args.system_tick,
              common_args.theme_color);

    web_applet_version = WebAppletVersion{common_args.library_version};

    const auto web_arg_storage = PopInData();
    ASSERT(web_arg_storage != nullptr);

    const auto& web_arg = web_arg_storage->GetData();
    ASSERT_OR_EXECUTE(web_arg.size() >= sizeof(WebArgHeader), { return; });

    web_arg_input_tlv_map = ReadWebArgs(web_arg, web_arg_header);

    LOG_DEBUG(Service_AM, "WebArgHeader: total_tlv_entries={}, shim_kind={}",
              web_arg_header.total_tlv_entries, web_arg_header.shim_kind);

    ExtractSharedFonts(system);

    switch (web_arg_header.shim_kind) {
    case ShimKind::Shop:
        InitializeShop();
        break;
    case ShimKind::Login:
        InitializeLogin();
        break;
    case ShimKind::Offline:
        InitializeOffline();
        break;
    case ShimKind::Share:
        InitializeShare();
        break;
    case ShimKind::Web:
        InitializeWeb();
        break;
    case ShimKind::Wifi:
        InitializeWifi();
        break;
    case ShimKind::Lobby:
        InitializeLobby();
        break;
    case ShimKind::Unknown8:
        LOG_WARNING(Service_AM, "(STUBBED) called, Unknown8 Applet is not implemented");
        break;
    default:
        ASSERT_MSG(false, "Invalid ShimKind={}", web_arg_header.shim_kind);
        break;
    }
}

Result WebBrowser::GetStatus() const {
    return status;
}

void WebBrowser::ExecuteInteractive() {
    const auto storage = PopInteractiveInData();

    if (!storage) {
        LOG_WARNING(Service_AM, "Received interactive data request with no data available");
        return;
    }

    const auto& data = storage->GetData();
    if (!web_session_enabled) {
        LOG_WARNING(Service_AM,
                    "Ignoring interactive web data for an applet without WebSessionEnabled");
        return;
    }

    if (data.size() < sizeof(WebSessionMessageHeader)) {
        LOG_WARNING(Service_AM, "Ignoring undersized WebSession message ({} bytes)", data.size());
        return;
    }

    WebSessionMessageHeader header;
    std::memcpy(&header, data.data(), sizeof(header));
    if (header.size + sizeof(header) != data.size()) {
        LOG_WARNING(Service_AM,
                    "Ignoring malformed WebSession message: kind={:#x}, payload_size={}, "
                    "storage_size={}",
                    header.kind, header.size, data.size());
        return;
    }

    const auto send_ack = [this, storage_size = static_cast<u32>(data.size())](
                              WebSessionReceiveMessageKind kind) {
        std::vector<u8> ack_data(sizeof(WebSessionMessageHeader) + 0xC);
        WebSessionMessageHeader ack_header;
        ack_header.kind = static_cast<u32>(kind);
        ack_header.size = 0xC;
        std::memcpy(ack_data.data(), &ack_header, sizeof(ack_header));
        std::memcpy(ack_data.data() + sizeof(ack_header), &storage_size, sizeof(storage_size));
        PushInteractiveOutData(std::make_shared<IStorage>(system, std::move(ack_data)));
    };

    switch (static_cast<WebSessionSendMessageKind>(header.kind)) {
    case WebSessionSendMessageKind::BrowserEngineContent: {
        std::string payload(data.begin() + sizeof(header), data.end());
        if (!payload.empty() && payload.back() == '\0') {
            payload.pop_back();
        }
        frontend.SendInteractiveData(std::move(payload));
        send_ack(WebSessionReceiveMessageKind::AckBrowserEngine);
        break;
    }
    case WebSessionSendMessageKind::SystemMessageAppear:
        // The frontend has already been opened by ExecuteOffline. Acknowledge the request so the
        // guest can continue its session startup instead of waiting for the hardware web applet.
        send_ack(WebSessionReceiveMessageKind::AckSystemMessage);
        break;
    case WebSessionSendMessageKind::Ack:
        // This is an acknowledgement for a browser-to-guest message and needs no frontend work.
        break;
    default:
        LOG_WARNING(Service_AM, "Ignoring unknown WebSession message kind {:#x}", header.kind);
        break;
    }
}

void WebBrowser::Execute() {
    // Checked here rather than per-frontend (e.g. previously only inside Qt's
    // GMainWindow::WebBrowserOpenWebPage, gated behind #ifdef CITRON_USE_QT_WEB_ENGINE) so the
    // behavior is identical for every frontend and independent of whether Qt WebEngine is even
    // compiled in - see the comment on Settings::values.disable_web_applet for why this exists
    // and why it defaults to off.
    if (Settings::values.disable_web_applet) {
        LOG_WARNING(Service_AM, "Web applet is disabled via settings, closing immediately");
        WebBrowserExit(WebExitReason::WindowClosed);
        return;
    }

    switch (web_arg_header.shim_kind) {
    case ShimKind::Shop:
        ExecuteShop();
        break;
    case ShimKind::Login:
        ExecuteLogin();
        break;
    case ShimKind::Offline:
        ExecuteOffline();
        break;
    case ShimKind::Share:
        ExecuteShare();
        break;
    case ShimKind::Web:
        ExecuteWeb();
        break;
    case ShimKind::Wifi:
        ExecuteWifi();
        break;
    case ShimKind::Lobby:
        ExecuteLobby();
        break;
    case ShimKind::Unknown8:
        WebBrowserExit(WebExitReason::EndButtonPressed);
        break;
    default:
        ASSERT_MSG(false, "Invalid ShimKind={}", web_arg_header.shim_kind);
        WebBrowserExit(WebExitReason::EndButtonPressed);
        break;
    }
}

void WebBrowser::ExtractOfflineRomFS() {
    LOG_DEBUG(Service_AM, "Extracting RomFS to {}",
              Common::FS::PathToUTF8String(offline_cache_dir));

    const auto extracted_romfs_dir = FileSys::ExtractRomFS(offline_romfs);

    const auto temp_dir = system.GetFilesystem()->CreateDirectory(
        Common::FS::PathToUTF8String(offline_cache_dir), FileSys::OpenMode::ReadWrite);

    FileSys::VfsRawCopyD(extracted_romfs_dir, temp_dir);

    // The page is now served exclusively from offline_cache_dir. Keeping the layered RomFS alive
    // would keep host handles open on its manual_html override files; ARCropolis rewrites those
    // files before opening its next page, and Windows then rejects the truncate/write operation.
    offline_romfs = nullptr;
}

void WebBrowser::WebBrowserExit(WebExitReason exit_reason, std::string last_url) {
    if ((web_arg_header.shim_kind == ShimKind::Share &&
         web_applet_version >= WebAppletVersion::Version196608) ||
        (web_arg_header.shim_kind == ShimKind::Web &&
         web_applet_version >= WebAppletVersion::Version524288)) {
        // TODO: Push Output TLVs instead of a WebCommonReturnValue
    }

    WebCommonReturnValue web_common_return_value;

    web_common_return_value.exit_reason = exit_reason;
    std::memcpy(&web_common_return_value.last_url, last_url.data(), last_url.size());
    web_common_return_value.last_url_size = last_url.size();

    LOG_DEBUG(Service_AM, "WebCommonReturnValue: exit_reason={}, last_url={}, last_url_size={}",
              exit_reason, last_url, last_url.size());

    complete = true;
    std::vector<u8> out_data(sizeof(WebCommonReturnValue));
    std::memcpy(out_data.data(), &web_common_return_value, out_data.size());
    PushOutData(std::make_shared<IStorage>(system, std::move(out_data)));
    Exit();
}

Result WebBrowser::RequestExit() {
    frontend.Close();
    R_SUCCEED();
}

bool WebBrowser::InputTLVExistsInMap(WebArgInputTLVType input_tlv_type) const {
    return web_arg_input_tlv_map.find(input_tlv_type) != web_arg_input_tlv_map.end();
}

std::optional<std::vector<u8>> WebBrowser::GetInputTLVData(WebArgInputTLVType input_tlv_type) {
    const auto map_it = web_arg_input_tlv_map.find(input_tlv_type);

    if (map_it == web_arg_input_tlv_map.end()) {
        return std::nullopt;
    }

    return map_it->second;
}

void WebBrowser::InitializeShop() {}

void WebBrowser::InitializeLogin() {}

void WebBrowser::InitializeOffline() {
    if (const auto session_flag = GetInputTLVData(WebArgInputTLVType::WebSessionEnabled);
        session_flag.has_value()) {
        web_session_enabled = !session_flag->empty() && session_flag->front() != 0;
    }

    const auto document_path =
        ParseStringValue(GetInputTLVData(WebArgInputTLVType::DocumentPath).value());

    const auto document_kind =
        ParseRawValue<DocumentKind>(GetInputTLVData(WebArgInputTLVType::DocumentKind).value());

    std::string additional_paths;

    switch (document_kind) {
    case DocumentKind::OfflineHtmlPage:
    default:
        title_id = system.GetApplicationProcessProgramID();
        nca_type = FileSys::ContentRecordType::HtmlDocument;
        additional_paths = "html-document";
        break;
    case DocumentKind::ApplicationLegalInformation:
        title_id = ParseRawValue<u64>(GetInputTLVData(WebArgInputTLVType::ApplicationID).value());
        nca_type = FileSys::ContentRecordType::LegalInformation;
        break;
    case DocumentKind::SystemDataPage:
        title_id = ParseRawValue<u64>(GetInputTLVData(WebArgInputTLVType::SystemDataID).value());
        nca_type = FileSys::ContentRecordType::Data;
        break;
    }

    static constexpr std::array<const char*, 3> RESOURCE_TYPES{
        "manual",
        "legal_information",
        "system_data",
    };

    offline_cache_dir = Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) /
                        fmt::format("offline_web_applet_{}/{:016X}",
                                    RESOURCE_TYPES[static_cast<u32>(document_kind) - 1], title_id);

    offline_document = Common::FS::ConcatPathSafe(
        offline_cache_dir, fmt::format("{}/{}", additional_paths, document_path));

    LOG_INFO(Service_AM, "Offline web session enabled: {}", web_session_enabled);

    // On hardware, manual_html is a LayeredFS overlay: files supplied by a mod replace matching
    // files from the title's HtmlDocument, while all other files continue to come from the base
    // document. ARCropolis intentionally supplies only its own files and relies on that fallback
    // for the shared help/ and common/ resources. Refresh our extracted cache when such an
    // overlay is present so ExecuteOffline extracts PatchManager's merged RomFS below instead of
    // serving a stale, standalone copy of the override page.
    if (document_kind == DocumentKind::OfflineHtmlPage) {
        const auto sd_mod_root =
            system.GetFileSystemController().GetSDMCModificationLoadRoot(title_id);

        if (sd_mod_root != nullptr) {
            const auto sd_override_document = Common::FS::ConcatPathSafe(
                std::filesystem::path{sd_mod_root->GetFullPath()},
                fmt::format("manual_html/{}/{}", additional_paths, document_path));

            if (Common::FS::Exists(sd_override_document)) {
                std::error_code error;
                std::filesystem::remove_all(offline_cache_dir, error);
                if (error) {
                    LOG_WARNING(Service_AM, "Failed to refresh offline web cache {}: {}",
                                Common::FS::PathToUTF8String(offline_cache_dir), error.message());
                }
            }
        }
    }
}

void WebBrowser::InitializeShare() {}

void WebBrowser::InitializeWeb() {
    external_url = ParseStringValue(GetInputTLVData(WebArgInputTLVType::InitialURL).value());

    // Resolve Nintendo CDN URLs.
    external_url = ResolveURL(external_url);
}

void WebBrowser::InitializeWifi() {}

void WebBrowser::InitializeLobby() {}

void WebBrowser::ExecuteShop() {
    LOG_WARNING(Service_AM, "(STUBBED) called, Shop Applet is not implemented");
    WebBrowserExit(WebExitReason::EndButtonPressed);
}

void WebBrowser::ExecuteLogin() {
    LOG_WARNING(Service_AM, "(STUBBED) called, Login Applet is not implemented");
    WebBrowserExit(WebExitReason::EndButtonPressed);
}

void WebBrowser::ExecuteOffline() {
    // Foreground "WebSession" applets (LibraryAppletMode::AllForegroundInitiallyHidden) are used
    // by both Super Mario 3D All-Stars and by SSBU mod frameworks such as ARCropolis (whose
    // mod-manager, workspace-selector, and config-editor UIs all open via skyline_web's
    // OfflineWebSession, which requests this exact mode). They are otherwise opened the same way
    // as a regular offline page below; what makes them a "session" is that they rely on the
    // interactive data channel below (see the interactive_data_callback passed to
    // OpenLocalWebPage, and ExecuteInteractive) for their actual back-and-forth instead of only a
    // one-shot final result. Previously this bailed out here without ever opening the page or
    // signaling completion, which left the calling guest thread blocked forever waiting on an
    // applet that had never actually started - i.e. a permanent hang the instant a session-mode
    // offline page was requested.

    const auto main_url = GetMainURL(Common::FS::PathToUTF8String(offline_document));

    if (!Common::FS::Exists(main_url)) {
        offline_romfs = GetOfflineRomFS(system, title_id, nca_type);

        if (offline_romfs == nullptr) {
            LOG_ERROR(Service_AM,
                      "RomFS with title_id={:016X} and nca_type={} cannot be extracted!", title_id,
                      nca_type);
            WebBrowserExit(WebExitReason::WindowClosed);
            return;
        }
    }

    LOG_INFO(Service_AM, "Opening offline document at {}",
             Common::FS::PathToUTF8String(offline_document));

    frontend.OpenLocalWebPage(
        Common::FS::PathToUTF8String(offline_document), [this] { ExtractOfflineRomFS(); },
        [this](WebExitReason exit_reason, std::string last_url) {
            WebBrowserExit(exit_reason, last_url);
        },
        [this](std::string data) {
            if (!web_session_enabled) {
                LOG_WARNING(Service_AM,
                            "Ignoring page message for an applet without WebSessionEnabled");
                return;
            }

            // WebSession content must be framed and NUL-terminated. The NNSDK client replaces
            // the last byte with a terminator when receiving it, so omitting it would truncate
            // the page's final character.
            WebSessionMessageHeader header;
            header.kind = static_cast<u32>(WebSessionReceiveMessageKind::BrowserEngineContent);
            header.size = static_cast<u32>(data.size() + 1);
            std::vector<u8> out_data(sizeof(header) + header.size);
            std::memcpy(out_data.data(), &header, sizeof(header));
            std::memcpy(out_data.data() + sizeof(header), data.data(), data.size());
            PushInteractiveOutData(std::make_shared<IStorage>(system, std::move(out_data)));
        });
}

void WebBrowser::ExecuteShare() {
    LOG_WARNING(Service_AM, "(STUBBED) called, Share Applet is not implemented");
    WebBrowserExit(WebExitReason::EndButtonPressed);
}

void WebBrowser::ExecuteWeb() {
    LOG_INFO(Service_AM, "Opening external URL at {}", external_url);

    frontend.OpenExternalWebPage(external_url,
                                 [this](WebExitReason exit_reason, std::string last_url) {
                                     WebBrowserExit(exit_reason, last_url);
                                 });
}

void WebBrowser::ExecuteWifi() {
    LOG_WARNING(Service_AM, "(STUBBED) called, Wifi Applet is not implemented");
    WebBrowserExit(WebExitReason::EndButtonPressed);
}

void WebBrowser::ExecuteLobby() {
    LOG_WARNING(Service_AM, "(STUBBED) called, Lobby Applet is not implemented");
    WebBrowserExit(WebExitReason::EndButtonPressed);
}
} // namespace Service::AM::Frontend
