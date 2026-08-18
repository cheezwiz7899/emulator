// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebKitGTK backend for the web browser applet. Alternative to QtNXWebEngineView
// (qt_web_browser.h) -- CPM/system-package policy and bundle size are why this exists
// at all, see CMakeModules/webview_native_deps.cmake.
//
// Method surface mirrors QtNXWebEngineView exactly, confirmed by
// `grep -n "web_applet->" src/citron/main.cpp` against the real tree, not guessed:
// LoadLocalWebPage, LoadExternalWebPage, IsFinished, SetFinished, GetExitReason,
// SetExitReason, GetLastURL, SetLastURL, GetCurrentURL, plus QWidget's own
// hide/move/resize/show/setFocus (free via inheritance below) and two new
// backend-agnostic methods -- EvaluateJavaScript, SetPageZoomFactor -- that replace
// the QtWebEngine-specific `web_applet->page()->runJavaScript(...)` /
// `web_applet->setZoomFactor(...)` call sites (main.cpp:947,983,1001,1079). Same two
// methods added as thin wrappers to QtNXWebEngineView so all backends share one call
// site -- see the main.cpp/qt_web_browser.h diff in this same patch.
//
// UNCOMPILED. No Qt6 + WebKitGTK combined toolchain in the sandbox this was written
// in. Bridge mechanism itself (script injection, message handlers, nav interception,
// eval) is the same logic already compiled and run in bridge_spike.c this session --
// reshaped into this class, not re-derived. The embedding piece
// (QWindow::fromWinId + createWindowContainer, X11-only) is new in this patch,
// unverified -- see the file's Embed() implementation for the honesty notes on that.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE
#include <QWidget>
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>
#endif

#include "core/frontend/applets/web_browser.h"

class GMainWindow;

namespace Core {
class System;
}

namespace InputCommon {
class InputSubsystem;
}

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE

// Package/registration-point wiring (QtWebBrowser's OpenLocalWebPage/
// OpenExternalWebPage/SendInteractiveData, all backend-agnostic signal emitters
// already) is UNCHANGED by this backend -- only the concrete view type GMainWindow
// constructs and drives changes, via the ActiveWebEngineView alias
// (main.h/main.cpp diff, same patch).
class WebKitGTKView : public QWidget {
public:
    explicit WebKitGTKView(GMainWindow& main_window, Core::System& system,
                           InputCommon::InputSubsystem* input_subsystem_);
    ~WebKitGTKView() override;

    void LoadLocalWebPage(const std::string& main_url, const std::string& additional_args);
    void LoadExternalWebPage(const std::string& main_url, const std::string& additional_args);

    [[nodiscard]] bool IsFinished() const { return finished; }
    void SetFinished(bool finished_) { finished = finished_; }

    [[nodiscard]] Service::AM::Frontend::WebExitReason GetExitReason() const { return exit_reason; }
    void SetExitReason(Service::AM::Frontend::WebExitReason exit_reason_) { exit_reason = exit_reason_; }

    [[nodiscard]] const std::string& GetLastURL() const { return last_url; }
    void SetLastURL(std::string last_url_) { last_url = std::move(last_url_); }

    // Backed by the decide-policy NAVIGATION_ACTION handler's captured URL --
    // ported equivalent of UrlRequestInterceptor::GetRequestedURL()
    // (util/url_request_interceptor.cpp), see bridge_spike.c's on_decide_policy for
    // the already-verified version of this exact logic.
    [[nodiscard]] QString GetCurrentURL() const;

    // Replaces web_applet->page()->runJavaScript(script, callback) uniformly across
    // all backends (main.cpp:983,1001,1079). Same signature/semantics as
    // QWebEnginePage::runJavaScript's 2-arg overload: fire-and-forget if callback
    // is null.
    void EvaluateJavaScript(const QString& script,
                            std::function<void(const QVariant&)> callback = {});

    // Replaces web_applet->setZoomFactor(...) (main.cpp:947). WebKitGTK equivalent:
    // webkit_web_view_set_zoom_level, confirmed present in the installed 2.52.3
    // headers used for bridge_spike.c this session.
    void SetPageZoomFactor(qreal factor);

    void hide();

private:
    // Real Qt mechanism for embedding a foreign native window as a QWidget,
    // confirmed via Qt 6.11/6.8 official "Window Embedding Example" docs this
    // session: QWindow::fromWinId(WId) + QWidget::createWindowContainer(QWindow*).
    // Confirmed native handle types per that same doc: HWND (Windows),
    // xcb_window_t (Linux), NSView (macOS) -- X11 only, no Wayland type listed.
    //
    // NOT VERIFIED ON WAYLAND. Checked for a Wayland path (xdg-foreign protocol,
    // Qt's Wayland plugin references it) and separately confirmed GTK itself has no
    // Wayland subsurface support (GNOME discourse, direct statement) -- real open
    // question, not resolved here. FallbackToTopLevelWindow() below is the
    // pragmatic non-blocking answer: detect X11 vs Wayland at runtime via
    // QGuiApplication::platformName(), embed on X11, fall back to a separate
    // top-level GtkWindow on Wayland (worse UX, ships, doesn't block on the harder
    // problem -- also literally what bridge_spike.c's spike already does, not
    // hidden as a coincidence).
    QWidget* Embed(QWidget* parent);
    void FallbackToTopLevelWindow();

    static void OnNxMessage(WebKitUserContentManager*, WebKitJavascriptResult*, gpointer);
    static void OnNxControl(WebKitUserContentManager*, WebKitJavascriptResult*, gpointer);
    static gboolean OnDecidePolicy(WebKitWebView*, WebKitPolicyDecision*,
                                   WebKitPolicyDecisionType, gpointer);

    GMainWindow& main_window;
    GtkWidget* gtk_window = nullptr; // only set if FallbackToTopLevelWindow() path taken
    WebKitWebView* webview = nullptr;
    QWidget* container = nullptr; // createWindowContainer() result, X11 path only

    Core::System& system;
    InputCommon::InputSubsystem* input_subsystem;

    std::atomic<bool> finished{false};
    Service::AM::Frontend::WebExitReason exit_reason{};
    std::string last_url;
    mutable QString requested_url; // set by OnDecidePolicy, read by GetCurrentURL
};

#endif // CITRON_USE_WEBKITGTK_WEB_ENGINE
