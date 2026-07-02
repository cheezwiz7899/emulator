#!/bin/sh
# SPDX-FileCopyrightText: 2026 citron Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
#
# package-citron-linux.sh — Build AppImage + tar.zst using pkgforge tooling.
#
# Called by build-citron-linux.sh's build_appimage_stage() after the install
# tree has been staged under build_dir/install-root.
# Can also be invoked directly if the install tree is already in place.
#
# Environment variables:
#   APP_VERSION   Version string embedded in artifact filenames (required)
#   ARCH          CPU architecture (default: uname -m)
#   ARCH_SUFFIX   Extra suffix appended to filenames (e.g. _v3)
#   DEVEL         Set to 'true' to rename app to "citron nightly" in AppImage
#   OUTPATH       Output directory for finished artifacts (default: $PWD/dist)
#   DESTDIR       Staging root prepended to /usr paths when locating the
#                 citron binary, desktop file, icon, and Qt translations
#                 (default: "" — real /usr, e.g. for an already-installed
#                 system citron). Set by build-citron-linux.sh's
#                 build_appimage_stage() to a build-tree staging directory so
#                 no system files are read from /usr and nothing is written
#                 there. Does NOT apply to libpulse/libgamemode — those are
#                 genuine host runtime libraries, not part of citron's install.
#   STRACE_MODE   Enable dlopen/LD_DEBUG scan when citron starts up (default: 0)
#                 Set to 1 only if xvfb-run is installed; see comment below.
#   DEPLOY_VULKAN Deploy Mesa/Vulkan DRI driver collection (default: 0)
#   DEPLOY_OPENGL Deploy OpenGL/EGL/GLX libs into AppImage (default: 0)
#   DEPLOY_PIPEWIRE Deploy PipeWire SPA/ALSA plugin trees (default: 0)
#   DEPLOY_GTK    Deploy GTK modules and gvfs into AppImage (default: 0)
#   CITRON_QT_PATH Qt6 installation prefix (= QT_TARGET_PATH from CMake,
#                 e.g. .../cpmcache/qt-bin/6.9.3/gcc_64). Set automatically
#                 by build-citron-linux.sh. Used to derive QT_LOCATION so
#                 quick-sharun bundles CPM Qt plugins, not system Qt plugins.

set -ex

ARCH="${ARCH:-$(uname -m)}"

if [ -z "${APP_VERSION:-}" ]; then
    echo "Error: APP_VERSION environment variable is not set." >&2
    exit 1
fi

DESTDIR="${DESTDIR:-}"
OUTNAME_BASE="citron_nightly-${APP_VERSION}-linux-${ARCH}${ARCH_SUFFIX:-}"
OUTPATH="${OUTPATH:-$PWD/dist}"
export OUTNAME="${OUTNAME_BASE}.AppImage"
export DESKTOP="${DESTDIR}/usr/share/applications/org.citron_emu.citron.desktop"

# Prefer a rasterised 256×256 PNG for .DirIcon — SVG fails silently as an
# AppImage icon on most desktop environments (appimaged, KDE, GNOME all expect
# a PNG for taskbar/launcher display).
#
# Priority:
#   1. Pre-built PNG installed by cmake (dist/org.citron_emu.citron.png →
#      share/icons/hicolor/256x256/apps/); commit this to the emulator repo
#      to make it permanent without needing any conversion tool.
#   2. Convert the installed SVG if a conversion tool is present.
#   3. Fall back to the raw SVG (icon will be broken on most desktops).
_png_from_repo="${DESTDIR}/usr/share/icons/hicolor/256x256/apps/org.citron_emu.citron.png"
_svg_icon="${DESTDIR}/usr/share/icons/hicolor/scalable/apps/org.citron_emu.citron.svg"
_png_icon="/tmp/citron-diricon-$$.png"
_try_svg2png() {
    if command -v rsvg-convert >/dev/null 2>&1; then
        rsvg-convert -w 256 -h 256 "$1" -o "$2" 2>/dev/null && return 0
    fi
    if command -v inkscape >/dev/null 2>&1; then
        inkscape --export-filename="$2" --export-width=256 --export-height=256 \
            "$1" >/dev/null 2>&1 && return 0
    fi
    if command -v convert >/dev/null 2>&1; then
        convert -background none -size 256x256 "$1" "$2" 2>/dev/null && return 0
    fi
    if command -v magick >/dev/null 2>&1; then
        magick -background none -size 256x256 "$1" "$2" 2>/dev/null && return 0
    fi
    return 1
}
if [ -f "$_png_from_repo" ]; then
    export ICON="$_png_from_repo"
elif [ -f "$_svg_icon" ] && _try_svg2png "$_svg_icon" "$_png_icon"; then
    export ICON="$_png_icon"
else
    export ICON="$_svg_icon"
fi

# ── quick-sharun deployment flags ───────────────────────────────────────────
#
# STRACE_MODE=0 — disable the dlopen / LD_DEBUG scan entirely.
#
#   In default mode (STRACE_MODE=1) quick-sharun spawns citron under
#   LD_DEBUG=libs for STRACE_TIME (default 5 s), then kills it, to capture
#   libs only dlopen'd at runtime.  When xvfb-run is NOT installed, citron
#   opens a real display window and the user must close it manually for the
#   build to continue.  Even with xvfb-run the scan is harmful for citron:
#   everything it captures is stuff we explicitly do NOT want bundled —
#   Vulkan ICD drivers (libvulkan_radeon.so, libvulkan_intel.so,
#   libLLVM.so.20.1 at 137 MB, …) loaded by vkCreateInstance(), the full
#   PipeWire SPA plugin tree loaded by PulseAudio's backend discovery, and
#   GTK accessibility modules from the GNOME a11y bus.  All of citron's
#   legitimate runtime deps are already covered by the static ldd scan
#   (explicit Qt plugins + libpulse + libgamemode).
export STRACE_MODE="${STRACE_MODE:-0}"
#
# DEPLOY_VULKAN=0 / DEPLOY_OPENGL=0 — do not stage Mesa DRI or the OpenGL set.
#   NOTE: DEPLOY_VULKAN=0 does NOT suppress Vulkan ICDs captured by the dlopen
#   scan (that requires STRACE_MODE=0 above); it only suppresses the separate
#   static DEPLOY_VULKAN staging block (libVkLayer_*.so, etc.).
#   libvulkan.so.1, libgbm.so, libdrm.so, libxcb-dri3.so, libxcb-glx.so are
#   all DT_NEEDED of citron / Qt plugins and are captured by ldd regardless.
#   SHARUN_ALLOW_SYS_VK_ICD=1 (set in .env below) makes sharun prefer the
#   host GPU ICD at runtime, so we never need to bundle GPU-vendor drivers.
export DEPLOY_VULKAN="${DEPLOY_VULKAN:-0}"
export DEPLOY_OPENGL="${DEPLOY_OPENGL:-0}"
#
# DEPLOY_PIPEWIRE=0 — suppress the PipeWire/SPA/ALSA plugin staging block.
#   On Ubuntu 24.04 PulseAudio is a PipeWire shim: ldd libpulse.so.0 shows
#   libpipewire-0.3.so.0.  quick-sharun's NEEDED_LIBS scan therefore sets
#   DEPLOY_PIPEWIRE=1, staging pipewire-0.3/* + spa-0.2/**/* + alsa-lib/
#   *pipewire* (~41 PipeWire plugins + 11 ALSA plugins + Bluetooth codecs).
#   These are backend-resolution plugins resolved from the HOST at runtime;
#   bundling them bloats the AppImage and causes host-version conflicts.
#   We carry libpulse.so.0 itself (passed explicitly below) — that is enough.
export DEPLOY_PIPEWIRE="${DEPLOY_PIPEWIRE:-0}"
#
# DEPLOY_GTK=0 — suppress GTK module / gvfs staging.
#   quick-sharun includes libqgtk3.so in the Qt plugin glob and runs ldd on
#   it; ldd sees libgtk-3.so.0, triggering DEPLOY_GTK=1 which stages
#   gtk-3.0/immodules/*, gvfs/libgvfscommon.so, and gio/modules/* (the full
#   GTK accessibility + filesystem integration stack).  quick-sharun already
#   sets QUICK_SHARUN_SKIP_DEPS_FOR="libqgtk3.so" so lib4bin skips transitive
#   deps of that plugin — but the DETECTION loop fires before SKIP applies and
#   sets DEPLOY_GTK=1 anyway.  The GTK platform theme should dlopen the HOST
#   GTK at runtime; bundling it causes version conflicts.
export DEPLOY_GTK="${DEPLOY_GTK:-0}"

# Tell quick-sharun to use CPM's Qt installation for plugins, not the SYSTEM
# Qt. Without this, quick-sharun defaults to $LIB_DIR/qt6/plugins (system Qt),
# whose shared libraries depend on the SYSTEM libQt6Core.so.6. That system
# version may be older than the CPM-built Qt 6.9.3 citron was compiled against,
# causing at AppImage runtime:
#   libQt6Core.so.6: version 'Qt_6.9' not found
#
# CITRON_QT_PATH = QT_TARGET_PATH from CMake = the Qt6 prefix directory (e.g.
# .../cpmcache/qt-bin/6.9.3/gcc_64). quick-sharun uses QT_LOCATION as the
# prefix and looks for plugins at $QT_LOCATION/plugins — that's exactly where
# aqt installs them.
if [ -n "${CITRON_QT_PATH:-}" ] && [ -z "${QT_LOCATION:-}" ]; then
    export QT_LOCATION="${CITRON_QT_PATH}"
fi

# Fix for "libQt6Core.so.6: version 'Qt_6.9' not found" at AppImage runtime.
#
# Root cause: quick-sharun COPIES each CPM Qt plugin into AppDir/shared/lib/
# before running ldd on the copy.  The plugins carry a $ORIGIN-relative
# DT_RUNPATH (e.g. "$ORIGIN/../../../lib") that is correct when the plugin is
# in $QT_LOCATION/plugins/platforms/, but resolves to a nonsense path once
# $ORIGIN becomes AppDir/shared/lib/.  ldd then falls through to the system
# linker search path and resolves libQt6Core.so.6 to the SYSTEM Qt (e.g.
# Qt 6.4.2 on Ubuntu 24.04).  That system copy is staged into the AppImage as
# AppDir/shared/lib/libQt6Core.so.6.  At runtime the CPM Qt 6.9 plugins
# require the Qt_6.9 version symbol — absent from the system 6.4 core — and
# the AppImage aborts with the version error.
#
# Fix: prepend CPM Qt's lib dir to LD_LIBRARY_PATH before calling quick-sharun.
# LD_LIBRARY_PATH has higher priority than DT_RUNPATH, so every ldd invocation
# inside lib4bin now finds CPM libQt6Core.so.6 (and all other Qt libs)
# regardless of broken $ORIGIN resolution.  The citron binary itself is
# unaffected: it carries DT_RPATH (set by _patch_binary_rpaths via
# patchelf --force-rpath), which has higher priority than LD_LIBRARY_PATH.
if [ -n "${CITRON_QT_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${CITRON_QT_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi


# Use the vendored quick-sharun.sh committed alongside this script rather than
# downloading from pkgforge HEAD.  The vendored copy has appimagetool pinned to
# 0.3.2 and sharun pinned to 1.0.0 (both already in the file); all DEPLOY_*
# overrides and cleanup rules in this script were calibrated against that
# specific version.  Pulling HEAD risks silent regressions if pkgforge changes
# DEPLOY_* defaults or the AppDir layout between releases.
_qs_src="$(cd "$(dirname "$0")" && pwd)/quick-sharun.sh"
if [ ! -f "${_qs_src}" ]; then
    printf 'ERROR: vendored quick-sharun.sh not found at %s\n' "${_qs_src}" >&2
    printf 'Commit quick-sharun.sh to the repo alongside this script.\n' >&2
    exit 1
fi
cp "${_qs_src}" ./quick-sharun
chmod +x ./quick-sharun

# Bundle binary + optional runtime libs.
#
# gamemode and libpulse are dlopen'd by cubeb/citron at runtime rather than
# linked at build time, so quick-sharun's automatic dependency scan won't
# find them — they must be passed explicitly if present.
#
# Locate them via `ldconfig -p` rather than a hardcoded /usr/lib/*.so* glob:
# on multiarch systems (Debian/Ubuntu) these libs live under
# /usr/lib/<triplet>/, not /usr/lib/ directly. A glob that matches nothing is
# passed to quick-sharun as a literal non-existent path by POSIX sh (no
# nullglob), which quick-sharun then treats as a hard requirement and aborts.
# Neither lib is guaranteed to be present (e.g. pure PipeWire setups have no
# libpulse.so), so both are best-effort.
EXTRA_LIBS=""

GAMEMODE_LIB="$(ldconfig -p 2>/dev/null | awk '/libgamemode\.so/ {print $NF; exit}')"
if [ -n "$GAMEMODE_LIB" ]; then
    EXTRA_LIBS="$EXTRA_LIBS $GAMEMODE_LIB"
fi

PULSE_LIB="$(ldconfig -p 2>/dev/null | awk '/libpulse\.so/ {print $NF; exit}')"
if [ -n "$PULSE_LIB" ]; then
    EXTRA_LIBS="$EXTRA_LIBS $PULSE_LIB"
fi

# shellcheck disable=SC2086
./quick-sharun "${DESTDIR}/usr/bin/citron"* $EXTRA_LIBS

# Qt translations: do NOT copy manually here.
# quick-sharun's --make-appimage (lib4bin) already copies
# /usr/share/qt6/translations → AppDir/lib/qt6/translations/ automatically
# when it detects Qt.  Adding a second copy from install-root would produce
# two separate translation trees (AppDir/usr/share/qt6/ and
# AppDir/lib/qt6/) wasting ~100 MB in the AppImage with no benefit.
# The qt.conf written below points Translations to the one quick-sharun stages.

# ── Post-dep-scan cleanup ────────────────────────────────────────────────────
# Belt-and-suspenders: remove any bloat that slipped past the DEPLOY_* flags
# or the STRACE_MODE=0 suppression (e.g. from ldd transitive chains we don't
# fully control, or from a future quick-sharun version that overrides our flags
# differently).  This runs BEFORE --make-appimage so lib4bin never sees them.
#
# Pattern                   Why excluded
# libvulkan_*.so            Mesa Vulkan ICD drivers: system-specific, huge.
#                           SHARUN_ALLOW_SYS_VK_ICD=1 uses host GPU driver.
# libLLVM*.so               Mesa LLVM backend (137 MB): only needed by Mesa ICDs.
# vulkan/icd.d/*            ICD manifest JSONs: useless without the ICD .so files.
# vulkan/implicit_layer.d/* Vulkan validation / overlay layers: dev tooling only.
# pipewire-*/               PipeWire backend plugins: resolved from host at runtime.
# spa-*/                    PipeWire SPA plugins: same.
# alsa-lib/                 ALSA plugins: same.
# gconv/                    glibc character-set converters: citron never calls iconv.
_appdir="${PWD}/AppDir"

# Explicitly stage ld-linux into AppDir/shared/lib/ — lib4bin's AppRun.lib
# prints "Interpreter not found!" and aborts if this file is absent.
#
# sharun -g (called by quick-sharun) is supposed to copy it, but it reads the
# interpreter path from AppDir/shared/bin/citron AFTER _patch_away_usr_lib_dir
# has run sed -i on the binary.  GNU sed on an ELF can silently corrupt the
# section that encodes PT_INTERP, causing sharun -g to find no interpreter and
# skip the copy.  We do it here from the original install-root binary (never
# touched by sed) so it is guaranteed present regardless of sharun -g's outcome.
_interp=$(patchelf --print-interpreter "${DESTDIR}/usr/bin/citron" 2>/dev/null || true)
if [ -z "${_interp}" ]; then
    # patchelf fallback: parse readelf output
    _interp=$(readelf -l "${DESTDIR}/usr/bin/citron" 2>/dev/null \
        | awk '/\[Requesting program interpreter:/{gsub(/[][]/,""); print $NF}')
fi
if [ -n "${_interp}" ] && [ -f "${_interp}" ]; then
    mkdir -p "${_appdir}/shared/lib"
    cp -L "${_interp}" "${_appdir}/shared/lib/${_interp##*/}"
    printf 'Staged interpreter: %s\n' "${_interp##*/}"
else
    printf 'WARNING: could not determine PT_INTERP of citron; AppImage may fail with "Interpreter not found!"\n' >&2
fi

find "${_appdir}" -name 'libvulkan_*.so'                    -delete 2>/dev/null || true
find "${_appdir}" -name 'libLLVM*.so*'                      -delete 2>/dev/null || true
find "${_appdir}" -path '*/vulkan/icd.d/*'                  -delete 2>/dev/null || true
find "${_appdir}" -path '*/vulkan/implicit_layer.d/*'       -delete 2>/dev/null || true
find "${_appdir}" -path '*/pipewire-*/*'          -type f   -delete 2>/dev/null || true
find "${_appdir}" -path '*/spa-*/*'               -type f   -delete 2>/dev/null || true
find "${_appdir}" -path '*/alsa-lib/*'            -type f   -delete 2>/dev/null || true
find "${_appdir}" -path '*/gconv/*'               -type f   -delete 2>/dev/null || true

# Remove system XCB libs that duplicate the CPM xcb-build copies.
#
# citron's DT_RPATH (set by _patch_binary_rpaths) points to xcb-build before
# any system path.  lib4bin strips those RPATHs, so at runtime all resolution
# goes through lib.path.  The Qt XCB plugin (libqxcb.so from CPM Qt) pulls in
# system libxcb.so.1, libXau.so.6, and libXdmcp.so.6 via its own ldd chain,
# staging them flat in AppDir/lib/.  The xcb-build copies of the same three
# libraries are already in AppDir at the absolute-mirrored path; keeping both
# is dead weight and violates the "CPM over system" policy.
#
# -maxdepth 1 on AppDir/lib/ hits only the flat system copies; the xcb-build
# copies are nested deep under home/thayne/…/xcb-build/lib/ and are untouched.
find "${_appdir}/lib" -maxdepth 1 -name 'libxcb.so*'    -delete 2>/dev/null || true
find "${_appdir}/lib" -maxdepth 1 -name 'libXau.so*'     -delete 2>/dev/null || true
find "${_appdir}/lib" -maxdepth 1 -name 'libXdmcp.so*'   -delete 2>/dev/null || true

# Qt Multimedia's FFmpeg backend — belt-and-suspenders removal.
#
# citron is built with -DCITRON_USE_BUNDLED_FFMPEG=ON (static FFmpeg for game
# media decoding) and -DCITRON_USE_QT_MULTIMEDIA=OFF (set in build-citron-linux.sh).
# With CITRON_USE_QT_MULTIMEDIA=OFF these files are never staged because Qt
# Multimedia is not linked.  These find/delete lines remain as a safety net in
# case the flag is ever changed or if a future Qt version starts staging the
# shared FFmpeg unconditionally.
#
# Background: Qt Multimedia pulls in a SEPARATE shared FFmpeg
# (libavcodec.so.61 ~80 MB, libavformat.so.61, libavutil.so.59, libswresample,
# libswscale) via libffmpegmediaplugin.so.  This is distinct from citron's own
# statically compiled FFmpeg and adds ~100 MB to the AppImage for no benefit
# (all camera/video recording code is Qt5-only and permanently dead under Qt6).
find "${_appdir}" -name 'libffmpegmediaplugin.so'   -delete 2>/dev/null || true
find "${_appdir}" -name 'libQt6FFmpegStub-*.so*'    -delete 2>/dev/null || true
find "${_appdir}" -name 'libavcodec.so*'             -delete 2>/dev/null || true
find "${_appdir}" -name 'libavformat.so*'            -delete 2>/dev/null || true
find "${_appdir}" -name 'libavutil.so*'              -delete 2>/dev/null || true
find "${_appdir}" -name 'libswresample.so*'          -delete 2>/dev/null || true
find "${_appdir}" -name 'libswscale.so*'             -delete 2>/dev/null || true



# Write a portable qt.conf next to libQt6Core.so.6 in shared/lib/.
#
# CPM Qt's libQt6Core.so.6 has a baked-in INSTALL_PREFIX of
# /home/<user>/cpmcache/qt-bin/6.9.x/gcc_64.  Without an explicit qt.conf,
# Qt uses that absolute prefix to locate its plugins and translations at
# runtime — which works on the build machine (the CPM directory exists on
# disk, outside the AppImage) but silently breaks on every other machine.
#
# A qt.conf placed next to libQt6Core.so.6 (AppDir/shared/lib/) overrides
# the embedded prefix.  All [Paths] entries are relative to Prefix; Prefix
# itself is relative to the qt.conf file's directory.
#
#   qt.conf at:   AppDir/shared/lib/qt.conf
#   Prefix = ..   →  AppDir/shared/
#   Plugins = lib →  AppDir/shared/lib/      (flat dir where lib4bin stages all plugins)
#   Translations  →  AppDir/usr/share/qt6/translations
#
# This also clobbers any qt.conf from the CPM installation that lib4bin or
# strace may have staged here, ensuring absolute build-machine paths cannot
# escape into the AppImage.
mkdir -p ./AppDir/shared/lib
cat > ./AppDir/shared/lib/qt.conf << 'QTCONF_EOF'
[Paths]
Prefix = ..
Plugins = lib
Imports = lib/qt6/qml
Qml2Imports = lib/qt6/qml
Translations = lib/qt6/translations
QTCONF_EOF

# Rename app in desktop file if building a devel/nightly AppImage
if [ "${DEVEL:-false}" = 'true' ]; then
    sed -i 's|^Name=citron$|Name=citron nightly|' ./AppDir/*.desktop 2>/dev/null || true
fi

# Allow system Vulkan ICD to override the bundled one at runtime, and write
# PGO profile data next to the running AppImage on exit (matches the old
# linuxdeploy $APPIMAGE_DIR convention). $APPIMAGE is exported by sharun's
# AppRun at runtime and points to the AppImage file's own path.
{
    printf 'SHARUN_ALLOW_SYS_VK_ICD=1\n'
    printf 'LLVM_PROFILE_FILE=$(dirname "$APPIMAGE")/default-%%p.profraw\n'
} > ./AppDir/.env

# Build the AppImage
./quick-sharun --make-appimage

mkdir -p "${OUTPATH}"
mv -v ./*.AppImage "${OUTPATH}/" 2>/dev/null || true
mv -v ./*.AppImage.* "${OUTPATH}/" 2>/dev/null || true

# Pack the portable tar.zst alongside the AppImage — the "+ tar.zst" half
# promised by this script's header comment, previously unimplemented.
#
# AppDir at this point is the exact same fully-debloated tree that was just
# squashed into the AppImage above (every DEPLOY_*/STRACE_MODE suppression and
# the post-dep-scan find/-delete block above already ran before
# --make-appimage), so this is a complete, AppRun-runnable, install-free
# alternative to the AppImage — not a partial copy. (AppDir/usr/ is not used
# by this layout — translations land under AppDir/lib/qt6/, see comment above —
# so packing only "usr", as the pre-quick-sharun pipeline did, would produce a
# near-empty archive here.)
#
# AppDir is then removed. It has no further purpose once packed, and leaving
# it sitting in OUTPATH means anything that uploads/copies OUTPATH wholesale
# (e.g. a CI artifact upload step) drags along every loose unpacked library a
# second time alongside the AppImage that already contains them compressed.
if [ -d ./AppDir ]; then
    if ! command -v zstd >/dev/null 2>&1; then
        printf 'ERROR: zstd is not installed — cannot produce %s.tar.zst\n' "${OUTNAME_BASE}" >&2
        printf 'Install zstd (e.g. apt-get install zstd) and re-run.\n' >&2
        exit 1
    fi
    tar -c --zstd -f "${OUTPATH}/${OUTNAME_BASE}.tar.zst" -C ./AppDir .
    rm -rf ./AppDir
fi

# Clean up build-tool byproducts that land in this same directory (OUTPATH),
# not just AppDir. build-citron-linux.sh sets OUTPATH to the same working
# directory this script cd's into and downloads tooling into, so anything
# these tools write to cwd ends up sitting next to the final artifacts unless
# removed explicitly:
#
#   quick-sharun — the tool itself, copied to ./quick-sharun near the top of
#                   this script (see the vendoring comment above). It has no
#                   purpose after packaging finishes.
#   appinfo      — a debug/metadata text file written to cwd by the
#                   underlying appimagetool binary during --make-appimage.
#                   Not generated by quick-sharun.sh itself, not meant for
#                   redistribution — internal build debug output only.
rm -f ./quick-sharun ./appinfo

echo "Artifacts in: ${OUTPATH}"
