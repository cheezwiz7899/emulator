// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebView2 backend for the web browser applet. Alternative to QtNXWebEngineView.
// Same method-surface-mirroring rationale as webkitgtk_web_browser.h -- see that
// file's header, not repeating it here.
//
// Embedding is structurally SIMPLER than the Linux side: WebView2's
// CreateCoreWebView2Controller takes a parent HWND directly and creates its own
// child HWND under it -- no foreign-window-pulling via QWindow::fromWinId needed.
// Force a real native HWND out of a plain QWidget via winId() (well-documented Qt
// behavior), hand that to WebView2, keep ICoreWebView2Controller::put_Bounds synced
// to the widget's resize/move events. No X11-vs-Wayland-shaped open question here --
// HWND is a single, stable concept on this platform.
//
// UNCOMPILED. No Windows/MSVC/clang-cl/WebView2 SDK toolchain in the sandbox this
// was written in -- see webview2_bridge.cpp (this session, earlier) for the same
// disclaimer, this file supersedes that one's scope-only draft with the real class
// shape but carries the same unverified-by-compilation status forward.

#pragma once

#include <atomic>
#include <string>

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE
#include <QWidget>
#include <wil/com.h>
#include <wrl.h>
#include <WebView2.h>
#endif

#include "core/frontend/applets/web_browser.h"

class GMainWindow;

namespace Core {
class System;
}

namespace InputCommon {
class InputSubsystem;
}

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE

class WebView2View : public QWidget {
public:
    explicit WebView2View(GMainWindow& main_window, Core::System& system, InputCommon::InputSubsystem* input_subsystem_);
    ~WebView2View() override;

    void LoadLocalWebPage(const std::string& main_url, const std::string& additional_args);
    void LoadExternalWebPage(const std::string& main_url, const std::string& additional_args);

    [[nodiscard]] bool IsFinished() const { return finished; }
    void SetFinished(bool finished_) { finished = finished_; }

    [[nodiscard]] Service::AM::Frontend::WebExitReason GetExitReason() const { return exit_reason; }
    void SetExitReason(Service::AM::Frontend::WebExitReason exit_reason_) { exit_reason = exit_reason_; }

    [[nodiscard]] const std::string& GetLastURL() const { return last_url; }
    void SetLastURL(std::string last_url_) { last_url = std::move(last_url_); }

    [[nodiscard]] QString GetCurrentURL() const { return requested_url; }

    // Replaces web_applet->page()->runJavaScript(...) uniformly across backends.
    void EvaluateJavaScript(const QString& script, std::function<void(const QVariant&)> callback = {});

    // Replaces web_applet->setZoomFactor(...). WebView2 equivalent:
    // ICoreWebView2Controller::put_ZoomFactor.
    void SetPageZoomFactor(qreal factor);

    void hide();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void moveEvent(QMoveEvent* event) override;

private:
    void InitWebView2(); // async CreateCoreWebView2EnvironmentWithOptions chain
    void SyncBounds();   // ICoreWebView2Controller::put_Bounds <- this widget's geometry

    HRESULT OnWebMessageReceived(ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs*);
    HRESULT OnNavigationStarting(ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs*);

    // Same envelope-tagging design as nx_shim_webview2.js / webview2_bridge.cpp --
    // WebView2 has one JS->native channel, not per-name handlers like WebKitGTK.
    static std::wstring JsonEscapeString(const std::wstring& input);

    GMainWindow& main_window;
    wil::com_ptr<ICoreWebView2Environment> environment;
    wil::com_ptr<ICoreWebView2Controller> controller;
    wil::com_ptr<ICoreWebView2> webview;

    Core::System& system;
    InputCommon::InputSubsystem* input_subsystem;

    std::atomic<bool> finished{false};
    Service::AM::Frontend::WebExitReason exit_reason{};
    std::string last_url;
    QString requested_url;
};

#endif // CITRON_USE_WEBVIEW2_WEB_ENGINE
