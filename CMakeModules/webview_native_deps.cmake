# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# CMakeModules/webview_native_deps.cmake — dependencies for the native web-browser
# applet backends (WebKitGTK/Linux, WebView2/Windows).
#
# Called from CMakeModules/dependencies.cmake, same as every other CPM package in
# this project.
#
# ── Policy note, read before changing this file ───────────────────────────────
# dependencies.cmake's own header states the project policy plainly: "All external
# dependencies must be fetched and managed via CPM... No system packages are
# assumed or required." CITRON_CHECK_SUBMODULES and CITRON_USE_BUNDLED_VCPKG are
# force-disabled to enforce it.
#
# WebKitGTK is a deliberate, necessary exception to that stated policy, not an
# oversight: it is a full browser engine with its own enormous dependency graph
# (GTK, glib, ICU, a dozen+ codec/media libraries...). Source-building it as a CPM
# sub-project is not a reasonable ask of anyone's build — no project reasonably
# vendors WebKit's own build this way, including projects far larger than this one.
# It comes from the system package manager via pkg-config, same as it already does
# for the libwebkit2gtk-4.1-dev package this session's spike work (bridge_spike.c)
# was built and run against. Flagging this explicitly rather than silently
# violating the stated policy without saying so.
#
# wil (Windows) has no such excuse — it's a small, header-only library, exactly
# CPM's normal use case — and is fetched via CPM below like everything else.

# ── wil (Windows Implementation Library) ───────────────────────────────────────
# Needed by webview2_web_browser.cpp for wil::com_ptr / wil::unique_cotaskmem_string.
# Checked before drafting that file: grepped the repo for existing wil::/WRL::
# usage — zero hits, this is a genuinely new dependency, not already vendored
# under a different name.
if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE)
    if (NOT TARGET WIL::WIL)
        CPMAddPackage(
            NAME wil
            GITHUB_REPOSITORY microsoft/wil
            GIT_TAG 6f37040e1a13f6c1104f6e4c30d5c50fdca75b26 # v1.0.240803.1, pin same
                                                              # policy as this repo's
                                                              # other CPM pins (see
                                                              # fmt/lz4 above in
                                                              # dependencies.cmake) —
                                                              # exact commit for this
                                                              # tag, not re-verified
                                                              # against upstream from
                                                              # this sandbox (no
                                                              # github.com network
                                                              # path to microsoft/wil
                                                              # tags checked here,
                                                              # confirm before merge)
            OPTIONS "WIL_BUILD_TESTS OFF" "WIL_BUILD_PACKAGING OFF"
        )
    endif()
endif()

# ── WebView2 SDK (Windows) ──────────────────────────────────────────────────────
# Normally distributed via NuGet (Microsoft.Web.WebView2). A .nupkg is a plain zip,
# so CPMAddPackage's URL form can fetch it directly without needing a NuGet client
# in the build — same "download an archive, don't source-build it" shape as this
# project's own Boost fetch above (URL form, not GIT_TAG form, because Boost's
# releases aren't consumed as a buildable CMake project either).
#
# NOT VERIFIED from this sandbox: nuget.org isn't in this environment's allowed
# network domains, so the exact URL/version/hash below could not be fetched and
# checked here. Confirm the package version and that CPM's extraction lands the
# expected build/native/include and build/native/<arch> layout before relying on
# this — flagging the gap rather than presenting an untested URL as proven.
if (WIN32 AND CITRON_USE_WEBVIEW2_WEB_ENGINE)
    if (NOT TARGET WebView2::WebView2)
        CPMAddPackage(
            NAME WebView2
            URL "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/1.0.2957.106"
            DOWNLOAD_ONLY YES # it's a NuGet layout (build/native/...), not a CMake
                              # project — same DOWNLOAD_ONLY pattern this file would
                              # need for any non-CMake archive dependency
        )
        if (WebView2_ADDED)
            add_library(WebView2::WebView2 INTERFACE IMPORTED)
            target_include_directories(WebView2::WebView2 INTERFACE
                "${WebView2_SOURCE_DIR}/build/native/include")
            # arch subdir (x64/arm64) and .lib name not confirmed against a real
            # extracted package from this sandbox — see the NOT VERIFIED note above.
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
    # Callers link PkgConfig::WEBKITGTK and PkgConfig::GTK3 — matches the target
    # names pkg_check_modules(... IMPORTED_TARGET ...) generates, no extra
    # aliasing needed.
endif()
