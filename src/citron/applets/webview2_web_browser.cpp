// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron/applets/webview2_web_browser.h"

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE

#include <QResizeEvent>
#include <QMoveEvent>

#include "citron/applets/webview2_web_browser_scripts.h"
#include "citron/main.h"

using Microsoft::WRL::Callback;

WebView2View::WebView2View(GMainWindow& main_window_, Core::System& system_,
                           InputCommon::InputSubsystem* input_subsystem_)
    : QWidget(&main_window_), main_window(main_window_), system(system_),
      input_subsystem(input_subsystem_) {
    // Forces Qt to create a real native HWND backing this widget right now rather
    // than lazily on first show() -- WebView2 needs a live parent HWND to create
    // its controller against. Well-documented QWidget::winId() behavior, not a
    // workaround.
    winId();
    InitWebView2();
}

WebView2View::~WebView2View() = default;

void WebView2View::InitWebView2() {
    // CreateCoreWebView2EnvironmentWithOptions -> environment created callback ->
    // CreateCoreWebView2Controller -> controller created callback -> get_CoreWebView2
    // -> wire handlers. Confirmed current signatures via Microsoft Learn this
    // session (webview2_bridge.cpp's research), reshaped into this class, not
    // re-derived.
    CreateCoreWebView2EnvironmentWithOptions(
        nullptr, nullptr, nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result)) return result;
                environment = env;
                environment->CreateCoreWebView2Controller(
                    reinterpret_cast<HWND>(winId()),
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this](HRESULT ctrl_result, ICoreWebView2Controller* ctrl) -> HRESULT {
                            if (FAILED(ctrl_result)) return ctrl_result;
                            controller = ctrl;
                            controller->get_CoreWebView2(&webview);
                            SyncBounds();

                            // Same DMABUF-class caveat doesn't apply here (Windows,
                            // not Linux/WebKitGTK) -- no equivalent env var/gate
                            // needed on this backend, nothing carried over
                            // needlessly from the Linux side.

                            std::wstring shim_source(WEBVIEW2_NX_SCRIPT,
                                                     WEBVIEW2_NX_SCRIPT +
                                                         std::size(WEBVIEW2_NX_SCRIPT) - 1);
                            webview->AddScriptToExecuteOnDocumentCreated(
                                shim_source.c_str(),
                                Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
                                    [](HRESULT, PCWSTR) -> HRESULT { return S_OK; }).Get());

                            webview->add_WebMessageReceived(
                                Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [this](ICoreWebView2* sender,
                                          ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                        return OnWebMessageReceived(sender, args);
                                    }).Get(),
                                nullptr);

                            // Main-frame-only, confirmed via 3 independent MS Learn
                            // pages this session -- no equivalent gap to the Linux
                            // decide-policy NAVIGATION_ACTION ambiguity here.
                            webview->add_NavigationStarting(
                                Callback<ICoreWebView2NavigationStartingEventHandler>(
                                    [this](ICoreWebView2* sender,
                                          ICoreWebView2NavigationStartingEventArgs* args) -> HRESULT {
                                        return OnNavigationStarting(sender, args);
                                    }).Get(),
                                nullptr);
                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());
}

void WebView2View::SyncBounds() {
    if (!controller) return;
    RECT bounds;
    bounds.left = 0;
    bounds.top = 0;
    bounds.right = width();
    bounds.bottom = height();
    controller->put_Bounds(bounds);
}

void WebView2View::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    SyncBounds();
}

void WebView2View::moveEvent(QMoveEvent* event) {
    QWidget::moveEvent(event);
    SyncBounds(); // bounds are widget-relative, but re-sync defensively on move too
}

void WebView2View::LoadLocalWebPage(const std::string& main_url, const std::string& additional_args) {
    if (!webview) return;
    // Mirrors QtNXWebEngineView::LoadLocalWebPage's QUrl::fromLocalFile(...) +
    // additional_args shape.
    std::wstring wide_url(main_url.begin(), main_url.end()); // ASCII paths only,
                                                              // same scope limit as
                                                              // the rest of this
                                                              // session's drafts
    std::wstring uri = L"file:///" + wide_url +
                       std::wstring(additional_args.begin(), additional_args.end());
    webview->Navigate(uri.c_str());
}

void WebView2View::LoadExternalWebPage(const std::string& main_url, const std::string& additional_args) {
    if (!webview) return;
    std::wstring uri(main_url.begin(), main_url.end());
    uri += std::wstring(additional_args.begin(), additional_args.end());
    webview->Navigate(uri.c_str());
}

void WebView2View::EvaluateJavaScript(const QString& script, std::function<void(const QVariant&)> callback) {
    if (!webview) return;
    std::wstring wscript = script.toStdWString();
    webview->ExecuteScript(
        wscript.c_str(),
        Callback<ICoreWebView2ExecuteScriptCompletedHandler>(
            [callback](HRESULT, PCWSTR) -> HRESULT {
                // Same "nothing currently consumes the result" note as the Linux
                // EvaluateJavaScript -- callback invoked for completion timing only.
                if (callback) callback(QVariant());
                return S_OK;
            }).Get());
}

void WebView2View::SetPageZoomFactor(qreal factor) {
    if (controller) {
        controller->put_ZoomFactor(static_cast<double>(factor));
    }
}

void WebView2View::hide() {
    SetFinished(true);
    QWidget::hide();
}

HRESULT WebView2View::OnWebMessageReceived(ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* args) {
    wil::unique_cotaskmem_string message_raw;
    if (FAILED(args->TryGetWebMessageAsString(&message_raw))) {
        // Not a plain string -- either the endApplet control envelope or
        // unexpected. Real integration needs get_WebMessageAsJson + parse for the
        // __citron_control envelope, same as webview2_bridge.cpp's sketched-not-
        // wired note -- not duplicating that TODO here, same status.
        return S_OK;
    }
    std::wstring message(message_raw.get());
    // wstring -> UTF-8 std::string for ForwardWebBrowserInteractiveData's signature.
    // QString gives a correct, already-available UTF-16 -> UTF-8 conversion rather
    // than a hand-rolled one for this narrow purpose.
    std::string utf8_message = QString::fromStdWString(message).toStdString();
    main_window.ForwardWebBrowserInteractiveData(utf8_message);
    return S_OK;
}

HRESULT WebView2View::OnNavigationStarting(ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs* args) {
    wil::unique_cotaskmem_string uri;
    if (FAILED(args->get_Uri(&uri))) return E_FAIL;
    requested_url = QString::fromWCharArray(uri.get());
    if (requested_url.contains(QStringLiteral("localhost"))) {
        SetFinished(true);
        SetExitReason(Service::AM::Frontend::WebExitReason::CallbackURL);
        SetLastURL(requested_url.toStdString());
    }
    // Never args->put_Cancel(TRUE) -- observe-only, matches UrlRequestInterceptor
    // and the Linux OnDecidePolicy.
    return S_OK;
}

#endif // CITRON_USE_WEBVIEW2_WEB_ENGINE
