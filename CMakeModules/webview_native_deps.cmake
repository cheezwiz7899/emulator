# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# CMakeModules/webview_native_deps.cmake — dependencies for the native web-browser
# applet backends (WebKitGTK/Linux, WebView2/Windows).
#
# Included from CMakeLists.txt whenever either backend option is on -- NOT
# nested inside the CITRON_USE_CPM block. WebKitGTK's dependency (pkg-config)
# has no CPM involvement at all; the WebView2 SDK below checks CITRON_USE_CPM
# because CPMAddPackage isn't defined without it.
#
# WebKitGTK is a deliberate exception to this project's CPM-first dependency
# policy: it's a full browser engine with a large dependency graph and is not
# reasonable to source-build as a CPM sub-project. It comes from the system
# package manager via pkg-config. WebView2 is fetched via CPM.

if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE AND NOT CITRON_USE_CPM)
    message(FATAL_ERROR "CITRON_USE_WEBVIEW2_WEB_ENGINE needs the WebView2 SDK fetched via "
                        "CPM (CITRON_USE_CPM=ON) — no non-CPM path is implemented.")
endif()

# ── WebView2 SDK (Windows) ──────────────────────────────────────────────────────
# Distributed via NuGet (Microsoft.Web.WebView2). A .nupkg is a plain zip so
# CPMAddPackage's URL form can fetch it directly without a NuGet client.
#
if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE AND CITRON_USE_CPM)
    if (NOT TARGET WebView2::WebView2)
        CPMAddPackage(
            NAME WebView2
            URL "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/1.0.2957.106/microsoft.web.webview2.1.0.2957.106.nupkg"
            DOWNLOAD_ONLY YES # NuGet layout (build/native/...), not a CMake project
        )
        if (WebView2_ADDED)
            add_library(WebView2::WebView2 INTERFACE IMPORTED)
            target_include_directories(WebView2::WebView2 INTERFACE
                "${WebView2_SOURCE_DIR}/build/native/include")
            if (MINGW)
                # The WebView2 SDK spells this Windows SDK include EventToken.h,
                # while MinGW provides it as eventtoken.h on case-sensitive hosts.
                # Supply a target-private compatibility spelling for Linux-to-
                # Windows cross-compilation without changing the SDK archive.
                set(WEBVIEW2_MINGW_COMPAT_DIR
                    "${CMAKE_CURRENT_BINARY_DIR}/webview2_mingw_compat")
                file(MAKE_DIRECTORY "${WEBVIEW2_MINGW_COMPAT_DIR}")
                file(WRITE "${WEBVIEW2_MINGW_COMPAT_DIR}/EventToken.h"
                     "#pragma once\n#include_next <eventtoken.h>\n")
                # The WebView2 SDK includes the Windows SDK spelling WeakReference.h,
                # while MinGW provides the same declarations as weakreference.h.
                # This only affects case-sensitive cross-compilation hosts.
                file(WRITE "${WEBVIEW2_MINGW_COMPAT_DIR}/WeakReference.h"
                     "#pragma once\n#include_next <weakreference.h>\n")
                target_include_directories(WebView2::WebView2 INTERFACE
                    "${WEBVIEW2_MINGW_COMPAT_DIR}")
            endif()
            target_link_libraries(WebView2::WebView2 INTERFACE
                "${WebView2_SOURCE_DIR}/build/native/x64/WebView2LoaderStatic.lib")
        endif()
    endif()
endif()

# ── WebKitGTK (Linux) — system package, not CPM. See policy note above. ────────
if (UNIX AND NOT APPLE AND NOT ANDROID AND CITRON_USE_WEBKITGTK_WEB_ENGINE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(WEBKITGTK REQUIRED IMPORTED_TARGET webkit2gtk-4.1)
    pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)
endif()
