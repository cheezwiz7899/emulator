// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// v2: adds font loading/injection, user agent, persistent storage, input thread,
// focus-first-link, window-close-requested -- see header comment and
// webkitgtk_web_browser.h/.cpp for the full account of why this v2 pass exists.
//
// Engine-model adaptation worth stating plainly rather than leaving implicit:
// WebView2 has exactly ONE native script-injection point
// (AddScriptToExecuteOnDocumentCreated, fires very early). QtWebEngine has three
// distinct stages (DocumentCreation/DocumentReady/Deferred) that
// qt_web_browser.cpp actually uses for different scripts at different times. This
// file gets that 3-stage timing back by hooking add_NavigationCompleted and
// running the "should run after load" scripts (fonts, focus-first-link) via
// ExecuteScript from there instead of a second native injection point that
// doesn't exist on this platform. Same practical effect, different mechanism --
// same kind of adaptation as the Linux side's DOCUMENT_END-for-both approximation,
// not hidden as if it were a native 1:1 match.
//
// UNCOMPILED -- see this file's header for the confidence-tier note (sourced from
// docs, not compiled against the real SDK, unlike the Linux side this round).

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

// %1-style Qt-arg substitution doesn't exist for plain std::wstring -- the CSS
// script needs 7 positional substitutions, same job QString::arg() does on the
// Linux side. Minimal, only handles what NX_FONT_CSS actually needs (7 sequential
// %N placeholders), not a general templating engine.
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
    // Mirrors qt_web_browser.cpp:59-60's setPersistentStoragePath -- userDataFolder
    // is CreateCoreWebView2EnvironmentWithOptions's 2nd param, was nullptr
    // (WebView2's own default location) in the v1 draft. Real fix, not cosmetic:
    // a nullptr userDataFolder means profile data lands wherever WebView2's
    // default happens to be for this install rather than citron's own directory,
    // same class of gap as the storage-path TODO the Linux side still carries
    // (that one's still a real gap; this one's now closed).
    auto storage_dir = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "webview2");
    std::wstring user_data_folder(storage_dir.begin(), storage_dir.end()); // ASCII-path
                                                                           // scope limit,
                                                                           // same as
                                                                           // elsewhere
                                                                           // this session

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

                            // Mirrors qt_web_browser.cpp:94-102's
                            // windowCloseRequested (JS window.close()).
                            webview->add_WindowCloseRequested(
                                Callback<ICoreWebView2WindowCloseRequestedEventHandler>(
                                    [this](ICoreWebView2* sender, IUnknown* args) -> HRESULT {
                                        return OnWindowCloseRequested(sender, args);
                                    }).Get(),
                                nullptr);

                            // NavigationCompleted drives the "runs after load"
                            // scripts (fonts, focus-first-link) AND doubles as the
                            // re-run-on-every-navigation trigger, same combined
                            // role FrameChanged plays on the Qt/WebKitGTK sides --
                            // one hook instead of two since WebView2 doesn't
                            // distinguish "just navigated" from "frame changed"
                            // the way the interceptor-based platforms do.
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
    // window_nx + gamepad, constructor-time, mirrors qt_web_browser.cpp:62-81.
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
    // ICoreWebView2Settings2 "continues" ICoreWebView2Settings (COM interface
    // versioning) -- confirmed via Microsoft Learn this session, put_UserAgent
    // lives on the extended interface, not the base one.
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

    // Wrapped in a DOMContentLoaded listener -- approximates Qt's DocumentReady
    // injection-point timing, since AddScriptToExecuteOnDocumentCreated alone
    // fires too early (before the DOM this script walks even exists). Same
    // approximation reasoning as the Linux side's DOCUMENT_END choice, different
    // mechanism because this platform's injection API is single-stage.
    std::wstring wrapped_css =
        L"window.addEventListener('DOMContentLoaded', function() { "
        L"var s = document.createElement('style'); s.textContent = \"" +
        css_source + L"\"; document.head.appendChild(s); });";
    webview->AddScriptToExecuteOnDocumentCreated(
        wrapped_css.c_str(),
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [](HRESULT, PCWSTR) -> HRESULT { return S_OK; }).Get());

    // LOAD_NX_FONT itself runs via NavigationCompleted (see InitWebView2), not
    // injected here -- this method's remaining job is just the CSS half.
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
                // result_json is the JSON-encoded JS result (ExecuteScript's own
                // contract, confirmed via Microsoft Learn this session) --
                // handles the boolean/number/string cases the footer-callback
                // check actually needs. Not a general JSON parser.
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
    // Same shape/reasoning as WebKitGTKView::InputThreadLoop -- see that file's
    // comment, not repeating the full rationale here.
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
        // Control-envelope path -- was "sketched not wired" in v1, closing it now.
        wil::unique_cotaskmem_string json_raw;
        if (FAILED(args->get_WebMessageAsJson(&json_raw))) {
            return S_OK;
        }
        std::wstring json(json_raw.get());
        // Narrow, deliberate check: only ever looking for this exact citron-owned
        // sentinel shape, not general JSON parsing. See nx_shim_webview2.js for
        // why this is envelope-tagged at all (WebView2 has one JS->native
        // channel, unlike WebKitGTK's separate nxMessage/nxControl handlers).
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
    // Mirrors qt_web_browser.cpp:94-102. No URL-match guard available here either
    // (same note as WebKitGTKView::OnClose) -- this event is already scoped to
    // this specific ICoreWebView2 instance, same reasoning for why that's fine.
    SetFinished(true);
    SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
    return S_OK;
}

#endif // CITRON_USE_WEBVIEW2_WEB_ENGINE
