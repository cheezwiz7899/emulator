# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# CMakeModules/webview_native_deps.cmake — dependencies for the native web-browser
# applet backends (WebKitGTK/Linux, WebView2/Windows).
#
# Included from CMakeLists.txt whenever either backend option is on -- NOT
# nested inside the CITRON_USE_CPM block. WebKitGTK's dependency (pkg-config)
# has no CPM involvement at all; wil/WebView2 below check CITRON_USE_CPM
# themselves, individually, since CPMAddPackage isn't defined without it.
#
# WebKitGTK is a deliberate exception to this project's CPM-first dependency
# policy: it's a full browser engine with a large dependency graph and is not
# reasonable to source-build as a CPM sub-project. It comes from the system
# package manager via pkg-config. wil and WebView2 are fetched via CPM.

if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE AND NOT CITRON_USE_CPM)
    message(FATAL_ERROR "CITRON_USE_WEBVIEW2_WEB_ENGINE needs wil + the WebView2 SDK, "
                        "both fetched via CPM (CITRON_USE_CPM=ON) — no non-CPM path "
                        "for these two is implemented.")
endif()

# ── wil (Windows Implementation Library) ───────────────────────────────────────
# Needed by webview2_web_browser.cpp for wil::com_ptr / wil::unique_cotaskmem_string.
# Checked before drafting that file: grepped the repo for existing wil::/WRL::
# usage — zero hits, this is a genuinely new dependency, not already vendored
# under a different name.
if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE AND CITRON_USE_CPM)
    if (NOT TARGET WIL::WIL)
        CPMAddPackage(
            NAME wil
            GITHUB_REPOSITORY microsoft/wil
            GIT_TAG 6f37040e1a13f6c1104f6e4c30d5c50fdca75b26 # v1.0.240803.1
            OPTIONS "WIL_BUILD_TESTS OFF" "WIL_BUILD_PACKAGING OFF"
        )
    endif()
endif()

# ── WebView2 SDK (Windows) ──────────────────────────────────────────────────────
# Distributed via NuGet (Microsoft.Web.WebView2). A .nupkg is a plain zip so
# CPMAddPackage's URL form can fetch it directly without a NuGet client.
#
# NOTE: the URL/version/layout below has not been verified against a real extracted
# package. Confirm before relying on it.
if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE AND CITRON_USE_CPM)
    if (NOT TARGET WebView2::WebView2)
        CPMAddPackage(
            NAME WebView2
            URL "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.2957.106"
            DOWNLOAD_ONLY YES # NuGet layout (build/native/...), not a CMake project
        )
        if (WebView2_ADDED)
            add_library(WebView2::WebView2 INTERFACE IMPORTED)
            target_include_directories(WebView2::WebView2 INTERFACE
                "${WebView2_SOURCE_DIR}/build/native/include")
            # NOTE: arch subdir and .lib name not confirmed against a real extracted package.
            target_link_libraries(WebView2::WebView2 INTERFACE
                "${WebView2_SOURCE_DIR}/build/native/x64/WebView2LoaderStatic.lib")
        endif()
    endif()
endif()

# ── WebKitGTK (Linux) — system package, not CPM. See policy note above. ────────
if (UNIX AND NOT APPLE AND CITRON_USE_WEBKITGTK_WEB_ENGINE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WEBKITGTK REQUIRED IMPORTED_TARGET webkit2gtk-4.1)
    pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)
endif()
