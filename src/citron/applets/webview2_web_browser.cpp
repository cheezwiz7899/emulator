// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebView2 has one script-injection point (AddScriptToExecuteOnDocumentCreated).
// "Runs after load" scripts (fonts, focus-first-link) are handled via
// NavigationCompleted + ExecuteScript instead of a second native injection stage.

#include "citron/applets/webview2_web_browser.h"

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE

#include <QResizeEvent>
#include <QMoveEvent>

#include "citron/applets/webview2_web_browser_scripts.h"
#include "citron/main.h"
#include "common/fs/path_util.h"
#include "hid_core/frontend/input_interpreter.h"
#include "hid_core/hid_types.h"

using Microsoft::WRL::Callback;

namespace {

struct DomKey {
    const wchar_t* key;
    const wchar_t* code;
    int key_code;
};

constexpr DomKey HIDButtonToDomKey(Core::HID::NpadButton button) {
    switch (button) {
    case Core::HID::NpadButton::Left:
    case Core::HID::NpadButton::StickLLeft:
        return {L"ArrowLeft", L"ArrowLeft", 37};
    case Core::HID::NpadButton::Up:
    case Core::HID::NpadButton::StickLUp:
        return {L"ArrowUp", L"ArrowUp", 38};
    case Core::HID::NpadButton::Right:
    case Core::HID::NpadButton::StickLRight:
        return {L"ArrowRight", L"ArrowRight", 39};
    case Core::HID::NpadButton::Down:
    case Core::HID::NpadButton::StickLDown:
        return {L"ArrowDown", L"ArrowDown", 40};
    default:
        return {L"", L"", 0};
    }
}

// QString::arg()-style %N substitution for std::wstring -- handles the 7 positional
// placeholders needed by NX_FONT_CSS.
std::wstring SubstitutePlaceholders(std::wstring script, const std::vector<std::wstring>& args) {
    for (size_t i = 0; i < args.size(); i++) {
        std::wstring placeholder = L"%" + std::to_wstring(i + 1);
        size_t pos;
        while ((pos = script.find(placeholder)) != std::wstring::npos) {
            script.replace(pos, placeholder.size(), args[i]);
        }
    }
    return script;
}

} // namespace

WebView2View::WebView2View(GMainWindow& main_window_, Core::System& system_,
                           InputCommon::InputSubsystem* input_subsystem_)
    : QWidget(&main_window_), main_window(main_window_), system(system_),
      input_subsystem(input_subsystem_),
      input_interpreter(std::make_unique<InputInterpreter>(system_)) {
    winId();
    InitWebView2();
}

WebView2View::~WebView2View() {
    SetFinished(true);
    StopInputThread();
}

void WebView2View::InitWebView2() {
    // Store profile data in citron's cache dir, mirroring qt_web_browser.cpp:59-60.
    auto storage_dir = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "webview2");
    std::wstring user_data_folder(storage_dir.begin(), storage_dir.end()); // ASCII paths only

    CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data_folder.c_str(), nullptr,
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

                            InjectPersistentScripts();

                            webview->add_WebMessageReceived(
                                Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                                    [this](ICoreWebView2* sender,
                                          ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                        return OnWebMessageReceived(sender, args);
                                    }).Get(),
                                nullptr);

                            webview->add_NavigationStarting(
                                Callback<ICoreWebView2NavigationStartingEventHandler>(
                                    [this](ICoreWebView2* sender,
                                          ICoreWebView2NavigationStartingEventArgs* args) -> HRESULT {
                                        return OnNavigationStarting(sender, args);
                                    }).Get(),
                                nullptr);

                            // Mirrors qt_web_browser.cpp:94-102's windowCloseRequested.
                            webview->add_WindowCloseRequested(
                                Callback<ICoreWebView2WindowCloseRequestedEventHandler>(
                                    [this](ICoreWebView2* sender, IUnknown* args) -> HRESULT {
                                        return OnWindowCloseRequested(sender, args);
                                    }).Get(),
                                nullptr);

                            // NavigationCompleted runs "after load" scripts and
                            // re-runs LOAD_NX_FONT on every navigation.
                            webview->add_NavigationCompleted(
                                Callback<ICoreWebView2NavigationCompletedEventHandler>(
                                    [this](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs*) -> HRESULT {
                                        EvaluateJavaScript(QString::fromStdWString(WEBVIEW2_LOAD_NX_FONT));
                                        return S_OK;
                                    }).Get(),
                                nullptr);
                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());
}

void WebView2View::InjectPersistentScripts() {
    // window_nx + gamepad at document creation, mirrors qt_web_browser.cpp:62-81.
    webview->AddScriptToExecuteOnDocumentCreated(
        WEBVIEW2_NX_SCRIPT,
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [](HRESULT, PCWSTR) -> HRESULT { return S_OK; }).Get());
    webview->AddScriptToExecuteOnDocumentCreated(
        WEBVIEW2_GAMEPAD_SCRIPT,
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [](HRESULT, PCWSTR) -> HRESULT { return S_OK; }).Get());
}

void WebView2View::SetUserAgent(UserAgent user_agent) {
    const wchar_t* user_agent_str = L"WebApplet";
    switch (user_agent) {
    case UserAgent::WebApplet: user_agent_str = L"WebApplet"; break;
    case UserAgent::ShopN: user_agent_str = L"ShopN"; break;
    case UserAgent::LoginApplet: user_agent_str = L"LoginApplet"; break;
    case UserAgent::ShareApplet: user_agent_str = L"ShareApplet"; break;
    case UserAgent::LobbyApplet: user_agent_str = L"LobbyApplet"; break;
    case UserAgent::WifiWebAuthApplet: user_agent_str = L"WifiWebAuthApplet"; break;
    }
    std::wstring full_ua = std::wstring(L"Mozilla/5.0 (Nintendo Switch; ") + user_agent_str +
                           L") AppleWebKit/606.4 (KHTML, like Gecko) NF/6.0.1.15.4 "
                           L"NintendoBrowser/5.1.0.20389";

    if (!webview) return;
    wil::com_ptr<ICoreWebView2Settings> settings;
    webview->get_Settings(&settings);
    // put_UserAgent is on ICoreWebView2Settings2, not the base ICoreWebView2Settings.
    auto settings2 = settings.try_query<ICoreWebView2Settings2>();
    if (settings2) {
        settings2->put_UserAgent(full_ua.c_str());
    }
}

void WebView2View::LoadExtractedFonts() {
    auto fonts_dir_str = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "fonts/");
    std::wstring fonts_dir(fonts_dir_str.begin(), fonts_dir_str.end());

    std::wstring css_source = SubstitutePlaceholders(
        WEBVIEW2_NX_FONT_CSS, {fonts_dir + L"FontStandard.ttf", fonts_dir + L"FontChineseSimplified.ttf",
                               fonts_dir + L"FontExtendedChineseSimplified.ttf",
                               fonts_dir + L"FontChineseTraditional.ttf", fonts_dir + L"FontKorean.ttf",
                               fonts_dir + L"FontNintendoExtended.ttf",
                               fonts_dir + L"FontNintendoExtended2.ttf"});

    // Wrap in a DOMContentLoaded listener to approximate Qt's DocumentReady timing,
    // since AddScriptToExecuteOnDocumentCreated fires before the DOM exists.
    std::wstring wrapped_css =
        L"window.addEventListener('DOMContentLoaded', function() { "
        L"var s = document.createElement('style'); s.textContent = \"" +
        css_source + L"\"; document.head.appendChild(s); });";
    webview->AddScriptToExecuteOnDocumentCreated(
        wrapped_css.c_str(),
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [](HRESULT, PCWSTR) -> HRESULT { return S_OK; }).Get());

    // LOAD_NX_FONT runs via NavigationCompleted (see InitWebView2).
}

void WebView2View::FocusFirstLinkElement() {
    EvaluateJavaScript(QString::fromStdWString(WEBVIEW2_FOCUS_LINK_ELEMENT_SCRIPT));
}

void WebView2View::LoadLocalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = true;
    LoadExtractedFonts();
    SetUserAgent(UserAgent::WebApplet);
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

    if (!webview) return;
    std::wstring wide_url(main_url.begin(), main_url.end());
    std::wstring uri = L"file:///" + wide_url + std::wstring(additional_args.begin(), additional_args.end());
    webview->Navigate(uri.c_str());
}

void WebView2View::LoadExternalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = false;
    SetUserAgent(UserAgent::WebApplet);
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

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
            [callback](HRESULT, PCWSTR result_json) -> HRESULT {
                if (!callback) return S_OK;
        // result_json is JSON-encoded (ExecuteScript contract) -- handles the
        // boolean/number/string cases the footer-callback check needs.
        QString result = QString::fromWCharArray(result_json ? result_json : L"null");
                QVariant qvariant;
                if (result == QStringLiteral("true")) {
                    qvariant = QVariant(true);
                } else if (result == QStringLiteral("false")) {
                    qvariant = QVariant(false);
                } else if (result.startsWith(QStringLiteral("\"")) && result.endsWith(QStringLiteral("\""))) {
                    qvariant = QVariant(result.mid(1, result.length() - 2));
                } else {
                    bool ok = false;
                    double num = result.toDouble(&ok);
                    if (ok) qvariant = QVariant(num);
                }
                callback(qvariant);
                return S_OK;
            }).Get());
}

void WebView2View::SetPageZoomFactor(qreal factor) {
    if (controller) {
        controller->put_ZoomFactor(static_cast<double>(factor));
    }
}

void WebView2View::SendKeyEvent(const std::wstring& key, const std::wstring& code, int key_code) {
    std::wstring script =
        L"(function() { var el = document.activeElement || document.body; "
        L"var opts = { key: '" + key + L"', code: '" + code + L"', keyCode: " +
        std::to_wstring(key_code) + L", which: " + std::to_wstring(key_code) +
        L", bubbles: true, cancelable: true }; "
        L"el.dispatchEvent(new KeyboardEvent('keydown', opts)); "
        L"el.dispatchEvent(new KeyboardEvent('keyup', opts)); })();";
    EvaluateJavaScript(QString::fromStdWString(script));
}

void WebView2View::hide() {
    SetFinished(true);
    QWidget::hide();
}

void WebView2View::StartInputThread() {
    if (input_thread_running) return;
    input_thread_running = true;
    input_thread = std::thread(&WebView2View::InputThreadLoop, this);
}

void WebView2View::StopInputThread() {
    if (!input_thread_running) return;
    input_thread_running = false;
    if (input_thread.joinable()) {
        input_thread.join();
    }
}

void WebView2View::InputThreadLoop() {
    // Same shape as WebKitGTKView::InputThreadLoop.
    std::this_thread::sleep_for(std::chrono::seconds(1));

    while (input_thread_running) {
        input_interpreter->PollInput();

        using Core::HID::NpadButton;
        for (NpadButton button : {NpadButton::A, NpadButton::B, NpadButton::X, NpadButton::Y,
                                  NpadButton::L, NpadButton::R}) {
            if (!input_interpreter->IsButtonPressedOnce(button)) continue;

            int callback_index = -1;
            const wchar_t* fallback_key = nullptr;
            int fallback_code = 0;
            switch (button) {
            case NpadButton::A: callback_index = 0; fallback_key = L"a"; fallback_code = 65; break;
            case NpadButton::B: callback_index = 1; fallback_key = L"b"; fallback_code = 66; break;
            case NpadButton::X: callback_index = 2; fallback_key = L"x"; fallback_code = 88; break;
            case NpadButton::Y: callback_index = 3; fallback_key = L"y"; fallback_code = 89; break;
            case NpadButton::L: callback_index = 6; break;
            case NpadButton::R: callback_index = 7; break;
            default: break;
            }

            const QString check_script =
                QStringLiteral("citron_key_callbacks[%1] != null").arg(callback_index);
            EvaluateJavaScript(check_script, [this, callback_index, fallback_key,
                                              fallback_code](const QVariant& has_callback) {
                if (has_callback.toBool()) {
                    EvaluateJavaScript(
                        QStringLiteral("citron_key_callbacks[%1]();").arg(callback_index));
                } else if (fallback_key) {
                    std::wstring upper_key(fallback_key);
                    for (auto& c : upper_key) c = towupper(c);
                    SendKeyEvent(fallback_key, upper_key, fallback_code);
                }
            });
        }

        for (NpadButton button : {NpadButton::Left, NpadButton::Up, NpadButton::Right,
                                  NpadButton::Down, NpadButton::StickLLeft, NpadButton::StickLUp,
                                  NpadButton::StickLRight, NpadButton::StickLDown}) {
            const bool pressed_once = input_interpreter->IsButtonPressedOnce(button);
            const bool held = input_interpreter->IsButtonHeld(button);
            if (pressed_once || held) {
                const DomKey dom_key = HIDButtonToDomKey(button);
                if (dom_key.key_code != 0) {
                    SendKeyEvent(dom_key.key, dom_key.code, dom_key.key_code);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void WebView2View::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    SyncBounds();
}

void WebView2View::moveEvent(QMoveEvent* event) {
    QWidget::moveEvent(event);
    SyncBounds();
}

void WebView2View::SyncBounds() {
    if (!controller) return;
    RECT bounds{0, 0, width(), height()};
    controller->put_Bounds(bounds);
}

HRESULT WebView2View::OnWebMessageReceived(ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* args) {
    wil::unique_cotaskmem_string message_raw;
    if (FAILED(args->TryGetWebMessageAsString(&message_raw))) {
        // Control-envelope path for endApplet -- WebView2 has one JS->native channel,
        // so endApplet is tagged with __citron_control rather than a separate handler.
        wil::unique_cotaskmem_string json_raw;
        if (FAILED(args->get_WebMessageAsJson(&json_raw))) {
            return S_OK;
        }
        std::wstring json(json_raw.get());
        // Check for the citron-owned control sentinel before treating as a page message.
        if (json.find(L"__citron_control") != std::wstring::npos &&
            json.find(L"endApplet") != std::wstring::npos) {
            SetFinished(true);
            SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
        }
        return S_OK;
    }
    std::wstring message(message_raw.get());
    main_window.ForwardWebBrowserInteractiveData(QString::fromStdWString(message).toStdString());
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
    return S_OK;
}

HRESULT WebView2View::OnWindowCloseRequested(ICoreWebView2*, IUnknown*) {
    // Mirrors qt_web_browser.cpp:94-102. Event is scoped to this ICoreWebView2 instance.
    SetFinished(true);
    SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
    return S_OK;
}

#endif // CITRON_USE_WEBVIEW2_WEB_ENGINE
