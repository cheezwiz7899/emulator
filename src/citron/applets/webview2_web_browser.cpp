// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebView2 has one script-injection point (AddScriptToExecuteOnDocumentCreated).
// "Runs after load" scripts (fonts, focus-first-link) are handled via
// NavigationCompleted + ExecuteScript instead of a second native injection stage.

#include "citron/applets/webview2_web_browser.h"

#ifdef CITRON_USE_WEBVIEW2_WEB_ENGINE

#include <chrono>
#include <cwctype>
#include <vector>

#include <QResizeEvent>
#include <QMoveEvent>
#include <QMetaObject>
#include <QThread>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

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

// Byte-wise (begin(), end()) construction only works for ASCII; paths/URLs can
// be non-ASCII UTF-8. QString::fromStdString is UTF-8-aware in Qt6.
std::wstring Utf8ToWide(const std::string& utf8) {
    return QString::fromStdString(utf8).toStdWString();
}

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
// Single left-to-right pass -- never rescans inserted replacement text, so a
// FontUrl() containing "%2F"/"%20" etc. can't be mistaken for a placeholder
// or get substituted twice (finding #12).
std::wstring SubstitutePlaceholders(const std::wstring& script, const std::vector<std::wstring>& args) {
    std::wstring result;
    result.reserve(script.size());
    for (size_t i = 0; i < script.size();) {
        if (script[i] == L'%' && i + 1 < script.size() && iswdigit(script[i + 1])) {
            size_t digits_end = i + 1;
            while (digits_end < script.size() && iswdigit(script[digits_end])) {
                ++digits_end;
            }
            size_t n = std::stoul(script.substr(i + 1, digits_end - i - 1));
            if (n >= 1 && n <= args.size()) {
                result += args[n - 1];
                i = digits_end;
                continue;
            }
            // No matching arg -- leave the placeholder text unchanged.
        }
        result += script[i];
        ++i;
    }
    return result;
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
    *alive = false; // first: any in-flight environment/controller completion
                    // lambda bails immediately instead of touching `this`
    SetFinished(true);
    StopInputThread(); // joins the thread; any invokeMethod(this, ...) still
                       // queued is auto-dropped by Qt once `this` is gone
    if (controller) {
        controller->Close(); // documented clean-teardown call, before the
                             // wil::com_ptr members release via RAII below
    }
}

void WebView2View::InitWebView2() {
    // Store profile data in citron's cache dir, mirroring qt_web_browser.cpp:59-60.
    auto storage_dir = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "webview2");
    std::wstring user_data_folder = Utf8ToWide(storage_dir);

    HRESULT create_result = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, user_data_folder.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this, life = alive](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (!*life) return S_OK; // `this` may already be destroyed
                if (FAILED(result)) {
                    SetFinished(true);
                    SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
                    return result;
                }
                environment = env;
                environment->CreateCoreWebView2Controller(
                    reinterpret_cast<HWND>(winId()),
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [this, life](HRESULT ctrl_result, ICoreWebView2Controller* ctrl) -> HRESULT {
                            if (!*life) return S_OK;
                            if (FAILED(ctrl_result)) {
                                SetFinished(true);
                                SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
                                return ctrl_result;
                            }
                            controller = ctrl;
                            // get_CoreWebView2 can fail; unchecked, webview stays
                            // null and everything below dereferences it (finding #14).
                            HRESULT webview_result = controller->get_CoreWebView2(&webview);
                            if (FAILED(webview_result) || !webview) {
                                SetFinished(true);
                                SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
                                return FAILED(webview_result) ? webview_result : E_FAIL;
                            }
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
                                        EvaluateJavaScript(QString::fromUtf8(WEB_BROWSER_LOAD_NX_FONT));
                                        FocusFirstLinkElement();
                                        return S_OK;
                                    }).Get(),
                                nullptr);
                            // Not called here -- InjectPersistentScripts's own
                            // completions call it once their counter hits 0
                            // (see below); calling it unconditionally here raced
                            // ahead of those registrations (finding #11).
                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());

    if (FAILED(create_result)) {
        SetFinished(true);
        SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
    }
}

void WebView2View::InjectPersistentScripts() {
    // window_nx + gamepad at document creation, mirrors qt_web_browser.cpp:62-81.
    // Both registrations are async; Navigate() must not fire until WebView2 has
    // actually confirmed both are registered, or the first page load could run
    // with no nx bridge at all. FlushPendingNavigation is deferred here instead
    // of being called unconditionally right after InitWebView2's setup.
    // Additive, not overwrite: FlushPendingNavigation (for is_local) queues
    // one more registration of its own on this same counter (finding #11).
    pending_script_registrations += 2;
    webview->AddScriptToExecuteOnDocumentCreated(
        WEBVIEW2_NX_SCRIPT,
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [this](HRESULT, PCWSTR) -> HRESULT {
                if (--pending_script_registrations == 0) FlushPendingNavigation();
                return S_OK;
            }).Get());
    webview->AddScriptToExecuteOnDocumentCreated(
        Utf8ToWide(WEB_BROWSER_GAMEPAD_SCRIPT).c_str(),
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [this](HRESULT, PCWSTR) -> HRESULT {
                if (--pending_script_registrations == 0) FlushPendingNavigation();
                return S_OK;
            }).Get());
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
    if (fonts_injected) {
        // Already registered from an earlier call -- nothing new to wait on.
        webview->Navigate(pending_url.c_str());
        return;
    }
    fonts_injected = true;

    auto fonts_dir_str = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "fonts/");
    std::wstring fonts_dir = Utf8ToWide(fonts_dir_str);

    // QUrl::fromLocalFile produces a proper file:// URL (forward slashes,
    // percent-encoded) instead of a raw Windows path. Windows paths have
    // backslashes, which are meaningless in a CSS url() and -- since this gets
    // substituted into a JS template literal below -- percent-encoding also
    // protects against any backtick/${ in the path corrupting that literal.
    auto FontUrl = [&](const wchar_t* filename) {
        QString path = QString::fromStdWString(fonts_dir + filename);
        return QUrl::fromLocalFile(path).toString().toStdWString();
    };

    std::wstring css_source = SubstitutePlaceholders(
        Utf8ToWide(WEB_BROWSER_NX_FONT_CSS), {FontUrl(L"FontStandard.ttf"), FontUrl(L"FontChineseSimplified.ttf"),
                               FontUrl(L"FontExtendedChineseSimplified.ttf"),
                               FontUrl(L"FontChineseTraditional.ttf"), FontUrl(L"FontKorean.ttf"),
                               FontUrl(L"FontNintendoExtended.ttf"), FontUrl(L"FontNintendoExtended2.ttf")});

    // WEB_BROWSER_NX_FONT_CSS is already a complete self-invoking script (builds its
    // own <style> tag via a JS template literal) -- inject as-is, no wrapping.
    // Registration is async; Navigate() is deferred to this completion (see
    // FlushPendingNavigation) so the first page load doesn't race ahead of it.
    webview->AddScriptToExecuteOnDocumentCreated(
        css_source.c_str(),
        Callback<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
            [this](HRESULT, PCWSTR) -> HRESULT {
                if (--pending_script_registrations == 0) {
                    webview->Navigate(pending_url.c_str());
                }
                return S_OK;
            }).Get());

    // LOAD_NX_FONT / FocusFirstLinkElement run via NavigationCompleted (see InitWebView2).
}

void WebView2View::FocusFirstLinkElement() {
    EvaluateJavaScript(QString::fromUtf8(WEB_BROWSER_FOCUS_LINK_ELEMENT_SCRIPT));
}

void WebView2View::LoadLocalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = true;
    pending_user_agent = UserAgent::WebApplet; // applied in FlushPendingNavigation --
                                               // SetUserAgent silently no-ops if
                                               // called before webview exists
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

    QString local_url = QUrl::fromLocalFile(QString::fromStdString(main_url)).toString() +
                        QString::fromStdString(additional_args);
    pending_url = local_url.toStdWString();
    has_pending_navigation = true;
    if (webview) {
        FlushPendingNavigation();
    }
}

void WebView2View::LoadExternalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = false;
    pending_user_agent = UserAgent::WebApplet;
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

    pending_url = Utf8ToWide(main_url) + Utf8ToWide(additional_args);
    has_pending_navigation = true;
    if (webview) {
        FlushPendingNavigation();
    }
}

// Called immediately above if webview is already live, or from InitWebView2's
// controller-creation handler if LoadLocalWebPage/LoadExternalWebPage ran
// before the async WebView2 init finished (they'd otherwise be silently
// dropped -- init is async, callers can't be expected to wait for it).
void WebView2View::FlushPendingNavigation() {
    if (!has_pending_navigation || !webview) return;
    // Persistent scripts (window_nx/gamepad) not registered yet -- their own
    // completion handlers call back in here once the counter hits 0
    // (finding #11).
    if (pending_script_registrations > 0) return;
    has_pending_navigation = false;
    SetUserAgent(pending_user_agent);
    if (is_local) {
        pending_script_registrations += 1;
        LoadExtractedFonts(); // navigates from its own completion handler, or
                              // immediately if the script was already registered
    } else {
        webview->Navigate(pending_url.c_str());
    }
}

void WebView2View::EvaluateJavaScript(const QString& script, std::function<void(const QVariant&)> callback) {
    if (!webview) return;
    std::wstring wscript = script.toStdWString();
    webview->ExecuteScript(
        wscript.c_str(),
        Callback<ICoreWebView2ExecuteScriptCompletedHandler>(
            // life guard: mirrors the environment/controller completion lambdas
            // (finding #14/round 2) -- callback may itself capture `this`
            // (finding #15).
            [callback, life = alive](HRESULT, PCWSTR result_json) -> HRESULT {
                if (!*life || !callback) return S_OK;
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

void WebView2View::hideEvent(QHideEvent* event) {
    SetFinished(true);
    QWidget::hideEvent(event);
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
            QMetaObject::invokeMethod(this, [this, check_script, callback_index, fallback_key,
                                             fallback_code] {
                EvaluateJavaScript(check_script, [this, callback_index, fallback_key,
                                                  fallback_code](const QVariant& has_callback) {
                    if (has_callback.toBool()) {
                        EvaluateJavaScript(
                            QStringLiteral("citron_key_callbacks[%1]();").arg(callback_index));
                    } else if (fallback_key) {
                        std::wstring upper_key(fallback_key);
                        for (auto& c : upper_key) c = towupper(c);
                        SendKeyEvent(fallback_key, L"Key" + upper_key, fallback_code);
                    }
                });
            }, Qt::QueuedConnection);
        }

        for (NpadButton button : {NpadButton::Left, NpadButton::Up, NpadButton::Right,
                                  NpadButton::Down, NpadButton::StickLLeft, NpadButton::StickLUp,
                                  NpadButton::StickLRight, NpadButton::StickLDown}) {
            const bool pressed_once = input_interpreter->IsButtonPressedOnce(button);
            const bool held = input_interpreter->IsButtonHeld(button);
            if (pressed_once || held) {
                const DomKey dom_key = HIDButtonToDomKey(button);
                if (dom_key.key_code != 0) {
                    QMetaObject::invokeMethod(this, [this, dom_key] {
                        SendKeyEvent(dom_key.key, dom_key.code, dom_key.key_code);
                    }, Qt::QueuedConnection);
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
    const qreal dpr = devicePixelRatioF();
    RECT bounds{0, 0, static_cast<LONG>(width() * dpr), static_cast<LONG>(height() * dpr)};
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
        // Real JSON parse, not a substring check (finding #13) -- a page message
        // could legitimately contain the literal text "__citron_control"/"endApplet"
        // without being the control envelope.
        QString json = QString::fromWCharArray(json_raw.get());
        QJsonParseError parse_error;
        QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8(), &parse_error);
        if (parse_error.error == QJsonParseError::NoError && doc.isObject()) {
            QJsonObject obj = doc.object();
            if (obj.value(QStringLiteral("__citron_control")).toString() ==
                QStringLiteral("endApplet")) {
                SetFinished(true);
                SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
            }
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
    if (QUrl(requested_url).host() == QStringLiteral("localhost")) {
        SetFinished(true);
        SetExitReason(Service::AM::Frontend::WebExitReason::CallbackURL);
        SetLastURL(requested_url.toStdString());
        // Citron-internal exit signal, not real content -- cancel rather than
        // let WebView2 attempt a real connection right as the applet closes.
        args->put_Cancel(TRUE);
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
