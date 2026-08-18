// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebView2 backend, v2. Same v1->v2 gap-closing as webkitgtk_web_browser.h -- see
// that file's header for the full account of what was missing and why (fonts,
// UA, storage, input thread, focus-first-link, window-close). Applying the same
// fixes here since this file was written in the same incomplete first pass and
// almost certainly has the identical gaps, not a separate audit from scratch.
//
// UNCOMPILED, still -- no Windows/WebView2 SDK/wrl.h available in this Linux
// sandbox regardless of how much else got fixed this round. The WebKitGTK side
// got a real compile-check against real headers this session; this file did not
// and structurally can't here. Every API name/signature below is sourced from
// Microsoft Learn (this session's searches), not memory -- see the .cpp's own
// citations -- but "sourced from real docs" and "compiled against the real SDK"
// are different confidence tiers and this is still the former, not the latter.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE
#include <QWidget>
#include <wil/com.h>
#include <wrl.h>
#include <WebView2.h>
#endif

#include "core/frontend/applets/web_browser.h"

class GMainWindow;
class InputInterpreter;

namespace Core {
class System;
}

namespace InputCommon {
class InputSubsystem;
}

namespace Core::HID {
enum class NpadButton : u64;
}

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE

class WebView2View : public QWidget {
public:
    enum class UserAgent {
        WebApplet,
        ShopN,
        LoginApplet,
        ShareApplet,
        LobbyApplet,
        WifiWebAuthApplet,
    };

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

    void EvaluateJavaScript(const QString& script, std::function<void(const QVariant&)> callback = {});
    void SetPageZoomFactor(qreal factor);

    void hide();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void moveEvent(QMoveEvent* event) override;

private:
    void InitWebView2();
    void SyncBounds();
    void SetUserAgent(UserAgent user_agent);
    void LoadExtractedFonts();
    void FocusFirstLinkElement();
    void InjectPersistentScripts(); // window_nx + gamepad, constructor-time, mirrors
                                    // qt_web_browser.cpp:62-81

    void StartInputThread();
    void StopInputThread();
    void InputThreadLoop();
    // Same deliberate deviation as WebKitGTKView::SendKeyEvent -- JS-synthesized
    // KeyboardEvent via ExecuteScript instead of native key injection. Unlike the
    // Linux side this isn't strictly forced by the embedding mechanism (WebView2's
    // own HWND is a real native child window with its own input focus, closer to
    // QtWebEngine's integration than WebKitGTK's foreign-window-pull), but kept
    // the same for both backends deliberately -- one mechanism to reason about and
    // test instead of two, and it's already proven to work via the bridge legs.
    void SendKeyEvent(const std::wstring& key, const std::wstring& code, int key_code);

    HRESULT OnWebMessageReceived(ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs*);
    HRESULT OnNavigationStarting(ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs*);
    HRESULT OnWindowCloseRequested(ICoreWebView2*, IUnknown*);

    static std::wstring JsonEscapeString(const std::wstring& input);

    GMainWindow& main_window;
    wil::com_ptr<ICoreWebView2Environment> environment;
    wil::com_ptr<ICoreWebView2Controller> controller;
    wil::com_ptr<ICoreWebView2> webview;

    Core::System& system;
    InputCommon::InputSubsystem* input_subsystem;
    std::unique_ptr<InputInterpreter> input_interpreter;
    std::thread input_thread;
    std::atomic<bool> input_thread_running{};

    bool is_local = false;
    std::atomic<bool> finished{false};
    Service::AM::Frontend::WebExitReason exit_reason{};
    std::string last_url;
    QString requested_url;
};

#endif // CITRON_USE_WEBVIEW2_WEB_ENGINE
