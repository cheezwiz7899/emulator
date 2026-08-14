# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
#
# CMakeModules/qt_download.cmake — Download Qt pre-built binaries via aqt
#
# Called from CMakeModules/dependencies.cmake when CITRON_USE_CPM=ON and ENABLE_QT=ON.
# Uses aqt (pip install aqtinstall) to fetch the correct Qt variant for the target
# platform.
#
# Target variants:
#   Windows llvm-mingw                       →  win64_llvm_mingw
#   Windows MSVC/clang-cl                    →  win64_msvc2022_64
#   Linux native x86-64                       →  linux_gcc_64 (via aqt)
#   Linux native aarch64                      →  linux_gcc_arm64 (host: linux_arm64)
#
# Cross-compilation (Linux host → Windows target):
#   QT_HOST_PATH is set to a Linux Qt install so moc/rcc/uic run on the host.
#   For native builds QT_HOST_PATH must NOT be set (it would trigger cross-compile mode).
#
# Prerequisites: Python3 + aqt must be installed (build script's ensure_aqt() handles this).

# The version is defined in dependencies.cmake as a CACHE variable.
if (NOT DEFINED CITRON_QT_VERSION)
    set(CITRON_QT_VERSION "6.9.3")
endif()

if (DEFINED ENV{CPM_SOURCE_CACHE})
    set(_DEFAULT_QT_BASE_DIR "$ENV{CPM_SOURCE_CACHE}/qt-bin")
elseif (DEFINED CPM_SOURCE_CACHE)
    set(_DEFAULT_QT_BASE_DIR "${CPM_SOURCE_CACHE}/qt-bin")
else()
    set(_DEFAULT_QT_BASE_DIR "${CMAKE_BINARY_DIR}/externals/qt-cpm")
endif()

set(CITRON_QT_BASE_DIR "${_DEFAULT_QT_BASE_DIR}" CACHE PATH
    "Base directory for aqt-managed Qt downloads")

# ── Find aqt ──────────────────────────────────────────────────────────────────
find_program(_AQT_EXECUTABLE NAMES aqt
    HINTS "$ENV{HOME}/.local/bin" "${CITRON_QT_BASE_DIR}")

if (NOT _AQT_EXECUTABLE)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if (Python3_FOUND)
        set(_AQT_EXECUTABLE "${Python3_EXECUTABLE}" "-m" "aqt")
    else()
        message(WARNING
            "[Qt] aqt not found and Python3 not available — Qt download skipped.\n"
            "     Pass -DQt6_DIR=... manually or run the build script first.")
        return()
    endif()
endif()

# ── Shared Linux host-arch selection ─────────────────────────────────────────
# aqt uses separate host-OS strings for x86-64 ("linux") and arm64
# ("linux_arm64"); the arch token is then linux_gcc_64 or linux_gcc_arm64.
# Computed once here and reused both by the native Linux target case below
# and by the cross-compile host Qt block, so moc/rcc/uic are always fetched
# for the actual build host architecture, not a hardcoded x86-64.
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_QT_HOST_OS       "linux_arm64")
    set(_QT_HOST_ARCH     "linux_gcc_arm64")
    set(_QT_HOST_DIR_NAME "gcc_arm64")
else()
    set(_QT_HOST_OS       "linux")
    set(_QT_HOST_ARCH     "linux_gcc_64")
    set(_QT_HOST_DIR_NAME "gcc_64")
endif()

# ── Determine target platform ──────────────────────────────────────────────────
# WIN32 is TRUE both for native MSYS2 builds and for Linux→Windows cross-compile
# because the CMAKE_SYSTEM_NAME is Windows in both cases.
#
# _QT_WEBENGINE_SUPPORTED tracks whether the qtwebengine module can actually be
# downloaded/used for the resolved target at all. Per Qt's own platform notes
# (https://doc.qt.io/qt-6/qtwebengine-platform-notes.html), WebEngine embeds
# Chromium, which currently does not compile with MinGW on Windows - this is a
# hard Chromium-level limitation, not something aqt or a different module
# selection can work around. It works for MSVC (and therefore clang-cl, which
# is MSVC-ABI-compatible and is what this project's Windows builds actually
# use it with), Linux, and macOS.
set(_QT_WEBENGINE_SUPPORTED TRUE)
if (WIN32)
    set(_QT_OS        "windows")
    set(_QT_TARGET    "desktop")
    if (MSVC)
        set(_QT_ARCH      "win64_msvc2022_64")
        set(_QT_DIR_NAME  "msvc2022_64")
    else()
        set(_QT_ARCH      "win64_llvm_mingw")
        set(_QT_DIR_NAME  "llvm-mingw_64")
        set(_QT_WEBENGINE_SUPPORTED FALSE)
    endif()
    set(_QT_CMAKE_SUB "lib/cmake/Qt6")
else()
    if (APPLE)
        set(_QT_OS        "mac")
        set(_QT_TARGET    "desktop")
        if (ARCHITECTURE_arm64)
            set(_QT_ARCH  "mac_arm64")
        else()
            set(_QT_ARCH  "mac_x64")
        endif()
        set(_QT_DIR_NAME  "macos")
        set(_QT_CMAKE_SUB "lib/cmake/Qt6")
    else()
        # Native Linux — reuse the shared host-arch selection above so
        # moc/rcc/uic (which run on the build host) are the correct ELF arch.
        set(_QT_OS        "${_QT_HOST_OS}")
        set(_QT_TARGET    "desktop")
        set(_QT_ARCH      "${_QT_HOST_ARCH}")
        set(_QT_DIR_NAME  "${_QT_HOST_DIR_NAME}")
        set(_QT_CMAKE_SUB "lib/cmake/Qt6")
    endif()
endif()

# ── Download target Qt ────────────────────────────────────────────────────────
# _QT_TARGET_DIR/_QT_TARGET_CMAKE are computed up front (not only inside the
# aqt-managed branch) so Qt6_DIR can be checked against the exact path aqt
# would manage, not just checked for existence. That distinction matters
# because Qt6_DIR is set as a FORCEd CACHE variable below: once a prior
# configure has populated it, a bare "is Qt6_DIR valid" check takes the
# shortcut on every later configure and never reaches the additional-modules
# check further down — so e.g. turning CITRON_USE_QT_WEB_ENGINE on in an
# existing build dir would silently never download qtwebengine. A genuinely
# external Qt6_DIR (system Qt, manual -DQt6_DIR=...) won't match this path
# and is still left untouched by the shortcut below.
set(_QT_TARGET_DIR   "${CITRON_QT_BASE_DIR}/${CITRON_QT_VERSION}/${_QT_DIR_NAME}")
set(_QT_TARGET_CMAKE "${_QT_TARGET_DIR}/${_QT_CMAKE_SUB}/Qt6Config.cmake")
get_filename_component(_QT_AQT_CMAKE_DIR "${_QT_TARGET_CMAKE}" DIRECTORY)

if (Qt6_DIR AND EXISTS "${Qt6_DIR}/Qt6Config.cmake" AND NOT ("${Qt6_DIR}" STREQUAL "${_QT_AQT_CMAKE_DIR}"))
    message(STATUS "[Qt] Using target Qt from Qt6_DIR: ${Qt6_DIR}")
    if (NOT QT_TARGET_PATH)
        get_filename_component(_tmp_path "${Qt6_DIR}/../../.." ABSOLUTE)
        set(QT_TARGET_PATH "${_tmp_path}" CACHE PATH "Path to Qt6 target root" FORCE)
    endif()
else()
    # Modules to request via aqt's -m flag.
    #
    # qtimageformats is a normal addon module for 6.9.3 on every target this
    # project builds for — confirmed directly against
    # `aqt list-qt <host> desktop --modules 6.9.3 <arch>` for win64_msvc2022_64,
    # win64_llvm_mingw, linux_gcc_64, and linux_gcc_arm64.
    #
    # qtsvg and qttools are NOT in any of those four module lists. Qt folded
    # QtSvg and the Qt6LinguistTools CMake package into the base/essentials
    # install for 6.9.x, so they're no longer separately installable addons
    # on any platform — requesting them via -m makes aqt fail outright
    # ("were not found while parsing XML of package information"), which is
    # what was actually failing, not a CMake-side bug. There's no renamed
    # equivalent to swap in; they just ship with the base install now. If a
    # future Qt/aqt release reintroduces them as separate addons, re-add
    # them here.
    #
    # Note: qtmultimedia is intentionally NOT downloaded here — it isn't used
    # by citron-neo on Qt6+.
    set(_QT_ADDL_MODULES qtimageformats)
    set(_QT_WEBENGINE_CMAKE "${_QT_TARGET_DIR}/lib/cmake/Qt6WebEngineCore/Qt6WebEngineCoreConfig.cmake")
    if (CITRON_USE_QT_WEB_ENGINE)
        if (_QT_WEBENGINE_SUPPORTED)
            list(APPEND _QT_ADDL_MODULES qtwebengine qtpositioning qtwebchannel)
        else()
            message(WARNING
                "[Qt] CITRON_USE_QT_WEB_ENGINE is ON, but qtwebengine is not available for "
                "${_QT_ARCH} (Qt WebEngine embeds Chromium, which does not compile with MinGW "
                "on Windows - this is a Qt/Chromium limitation, not an aqt packaging gap). "
                "Use the MSVC/clang-cl toolchain instead if you need the web applet frontend; "
                "the build will proceed without it and CITRON_USE_QT_WEB_ENGINE will have no "
                "effect for this target.")
        endif()
    endif()
    # Svg/LinguistTools ship with the base install now (see note above), so
    # these are no longer a gate for an additional aqt call — just a
    # post-install sanity check further down that the base install actually
    # delivered them.
    set(_QT_SVG_CMAKE  "${_QT_TARGET_DIR}/lib/cmake/Qt6Svg/Qt6SvgConfig.cmake")
    set(_QT_TOOL_CMAKE "${_QT_TARGET_DIR}/lib/cmake/Qt6LinguistTools/Qt6LinguistToolsConfig.cmake")

    if (NOT EXISTS "${_QT_TARGET_CMAKE}")
        # Fresh install: request the base and every additional module in a
        # single aqt invocation, instead of a base call followed by a
        # separate modules-only call into the same --outputdir. The
        # modules-only follow-up has been observed to fail outright right
        # after a CPM cache wipe — i.e. with nothing stale on disk, and with
        # the argv aqt actually receives confirmed correct (dumped and
        # checked directly) — which points at running install-qt a second
        # time against a directory the first call just populated, not at
        # the argument list. One call removes that second invocation
        # entirely for the common fresh-install case.
        message(STATUS "[Qt] Downloading Qt ${CITRON_QT_VERSION} ${_QT_ARCH} via aqt (modules: ${_QT_ADDL_MODULES})...")
        file(MAKE_DIRECTORY "${CITRON_QT_BASE_DIR}")

        execute_process(
            COMMAND ${_AQT_EXECUTABLE} install-qt
                    ${_QT_OS} ${_QT_TARGET}
                    ${CITRON_QT_VERSION} ${_QT_ARCH}
                    --outputdir "${CITRON_QT_BASE_DIR}"
                    --modules ${_QT_ADDL_MODULES}
            RESULT_VARIABLE _qt_result
            OUTPUT_VARIABLE _qt_output
            ERROR_VARIABLE  _qt_error
        )
        if (NOT _qt_result EQUAL 0)
            # FATAL_ERROR, not WARNING + return(): return() here only unwinds
            # this include()'d file — verified empirically that it does NOT
            # stop dependencies.cmake or CMakeLists.txt, both of which keep
            # running after their respective include() calls. With Qt6_DIR
            # left unset, a later find_package(Qt6 ...) elsewhere in the
            # project can silently succeed against whatever Qt6 happens to be
            # on CMAKE_PREFIX_PATH instead (e.g. an MSYS2 shell's own
            # /clang64 Qt6, built for the MinGW-w64 runtime) — the configure
            # "succeeds", moc/uic/compile all run against that Qt, and the
            # build only fails hundreds of steps later at link time with an
            # unrelated-looking error (missing mingw32.lib when linking an
            # MSVC-ABI/clang-cl+lld-link binary). Fail loud, immediately,
            # here instead.
            message(FATAL_ERROR
                "[Qt] aqt install failed (exit ${_qt_result}): ${_qt_error}\n"
                "     Pass -DQt6_DIR=... manually or ensure aqt is installed.")
        endif()
        message(STATUS "[Qt] Qt ${CITRON_QT_VERSION} target and additional modules downloaded")
    else()
        # Base already present (existing build dir, or Qt6_DIR pointed here
        # from an earlier configure — see the aqt-managed check above).
        # webengine is the only entry in _QT_ADDL_MODULES that's still a real
        # addon module aqt can top up after the fact (svg/tools ship with the
        # base install — see notes above, so there's nothing to top up there).
        if (CITRON_USE_QT_WEB_ENGINE AND _QT_WEBENGINE_SUPPORTED AND NOT EXISTS "${_QT_WEBENGINE_CMAKE}")
            message(STATUS "[Qt] Downloading Qt ${CITRON_QT_VERSION} additional modules (${_QT_ADDL_MODULES})...")
            execute_process(
                COMMAND ${_AQT_EXECUTABLE} install-qt
                        ${_QT_OS} ${_QT_TARGET}
                        ${CITRON_QT_VERSION} ${_QT_ARCH}
                        --outputdir "${CITRON_QT_BASE_DIR}"
                        --modules ${_QT_ADDL_MODULES}
                RESULT_VARIABLE _qt_addl_result
                OUTPUT_VARIABLE _qt_addl_output
                ERROR_VARIABLE  _qt_addl_error
            )
            if (NOT _qt_addl_result EQUAL 0)
                message(WARNING
                    "[Qt] Additional module install failed (exit ${_qt_addl_result}) for "
                    "(${_QT_ADDL_MODULES}): ${_qt_addl_error}\n"
                    "     Build may fail on missing Qt components.")
            endif()
        endif()
    endif()

    if (EXISTS "${_QT_TARGET_CMAKE}")
        get_filename_component(_qt6_dir "${_QT_TARGET_CMAKE}" DIRECTORY)
        set(Qt6_DIR "${_qt6_dir}" CACHE PATH "Path to Qt6Config.cmake (from aqt)" FORCE)
        set(QT_TARGET_PATH "${_QT_TARGET_DIR}" CACHE PATH "Path to Qt6 target root" FORCE)

        message(STATUS "[Qt] Qt6_DIR = ${Qt6_DIR}")
        message(STATUS "[Qt] QT_TARGET_PATH = ${QT_TARGET_PATH}")

        # Svg/LinguistTools are expected to ship with the base install (see
        # notes above) with no separate -m fallback if they're missing —
        # surface that clearly here rather than letting a much later,
        # unrelated-looking find_package(Qt6 COMPONENTS Svg) failure be the
        # first sign something's wrong.
        if (NOT EXISTS "${_QT_SVG_CMAKE}")
            message(WARNING
                "[Qt] Qt6Svg not found under ${_QT_TARGET_DIR} — expected it to ship with "
                "base Qt ${CITRON_QT_VERSION}; find_package(Qt6 COMPONENTS Svg) will fail.")
        endif()
        if (NOT EXISTS "${_QT_TOOL_CMAKE}")
            message(WARNING
                "[Qt] Qt6LinguistTools not found under ${_QT_TARGET_DIR} — expected it to ship "
                "with base Qt ${CITRON_QT_VERSION}.")
        endif()
    endif()
endif()

# Prepend target path so internal dependencies (like Qt6CoreTools) are found here first
if (QT_TARGET_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${QT_TARGET_PATH}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE PATH "Search path for Qt and other dependencies" FORCE)
endif()

# ── Host Qt for cross-compilation (Linux host → Windows target) ───────────────
# Only needed when the host OS differs from the target (CMAKE_CROSSCOMPILING=TRUE
# or when CMAKE_HOST_UNIX is TRUE but we're targeting WIN32).
# For native Linux builds: skip entirely — the target Qt IS the host Qt.
# QT_HOST_PATH must NOT be set for native builds (it triggers cross-compile mode).
if (CMAKE_HOST_UNIX AND WIN32)
    if (QT_HOST_PATH AND EXISTS "${QT_HOST_PATH}/lib/cmake/Qt6/Qt6Config.cmake")
        message(STATUS "[Qt] Using host Qt from QT_HOST_PATH: ${QT_HOST_PATH}")
    else()
        set(_QT_HOST_DIR   "${CITRON_QT_BASE_DIR}/${CITRON_QT_VERSION}/${_QT_HOST_DIR_NAME}")
        set(_QT_HOST_CMAKE "${_QT_HOST_DIR}/lib/cmake/Qt6/Qt6Config.cmake")

        if (NOT EXISTS "${_QT_HOST_CMAKE}")
            message(STATUS "[Qt] Downloading Qt ${CITRON_QT_VERSION} ${_QT_HOST_ARCH} host tools via aqt...")
            execute_process(
                COMMAND ${_AQT_EXECUTABLE} install-qt ${_QT_HOST_OS} desktop
                        ${CITRON_QT_VERSION} ${_QT_HOST_ARCH}
                        --outputdir "${CITRON_QT_BASE_DIR}"
                RESULT_VARIABLE _qt_host_result
                OUTPUT_QUIET ERROR_QUIET
            )
            if (NOT _qt_host_result EQUAL 0)
                message(WARNING "[Qt] Host Qt download failed — cross-compile may fail")
            endif()
        endif()

        if (EXISTS "${_QT_HOST_CMAKE}")
            set(QT_HOST_PATH "${_QT_HOST_DIR}" CACHE PATH "Host Qt for cross-compile tools" FORCE)
            message(STATUS "[Qt] QT_HOST_PATH = ${QT_HOST_PATH}")
        endif()
    endif()
endif()

# Prepend host path for cross-compile tool discovery
if (QT_HOST_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${QT_HOST_PATH}")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE PATH "Search path for Qt and other dependencies" FORCE)
endif()
