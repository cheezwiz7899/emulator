// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron/applets/webkitgtk_web_browser.h"

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE

#include <cstring>

#include <QGuiApplication>
#include <QTimer>
#include <QVBoxLayout>
#include <QWindow>

// GTK/WebKitGTK headers included here (not in the header) to avoid the
// signals/slots name collision. See webkitgtk_web_browser.h for details.
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>

#include <gdk/gdk.h>
#if defined(GDK_WINDOWING_X11)
#include <gdk/gdkx.h>
#endif

#include "citron/applets/webkitgtk_web_browser_scripts.h"
#include "citron/main.h"
#include "common/fs/path_util.h"
#include "hid_core/frontend/input_interpreter.h"
#include "hid_core/hid_types.h"

namespace {

// DOM key triple for SendKeyEvent's eval -- analogous to HIDButtonToKey in qt_web_browser.cpp.
struct DomKey {
    const char* key;
    const char* code;
    int key_code;
};

constexpr DomKey HIDButtonToDomKey(Core::HID::NpadButton button) {
    switch (button) {
    case Core::HID::NpadButton::Left:
    case Core::HID::NpadButton::StickLLeft:
        return {"ArrowLeft", "ArrowLeft", 37};
    case Core::HID::NpadButton::Up:
    case Core::HID::NpadButton::StickLUp:
        return {"ArrowUp", "ArrowUp", 38};
    case Core::HID::NpadButton::Right:
    case Core::HID::NpadButton::StickLRight:
        return {"ArrowRight", "ArrowRight", 39};
    case Core::HID::NpadButton::Down:
    case Core::HID::NpadButton::StickLDown:
        return {"ArrowDown", "ArrowDown", 40};
    default:
        return {"", "", 0};
    }
}

// Heap context for the GAsyncReadyCallback trampoline -- carries the std::function
// that a plain C function pointer can't capture directly.
struct EvalCallbackContext {
    std::function<void(const QVariant&)> callback;
};

void OnEvaluateJavaScriptFinished(GObject* source, GAsyncResult* result, gpointer user_data) {
    auto* ctx = static_cast<EvalCallbackContext*>(user_data);
    GError* error = nullptr;
    JSCValue* value =
        webkit_web_view_evaluate_javascript_finish(WEBKIT_WEB_VIEW(source), result, &error);

    QVariant qvariant;
    if (value) {
        if (jsc_value_is_boolean(value)) {
            qvariant = QVariant(static_cast<bool>(jsc_value_to_boolean(value)));
        } else if (jsc_value_is_number(value)) {
            qvariant = QVariant(jsc_value_to_double(value));
        } else if (jsc_value_is_string(value)) {
            char* str = jsc_value_to_string(value);
            qvariant = QVariant(QString::fromUtf8(str));
            g_free(str);
        }
        // Object/array/undefined/null JS types -> default-constructed QVariant; no
        // current call site needs them.
    } else if (error) {
        g_error_free(error);
    }

    if (ctx->callback) {
        ctx->callback(qvariant);
    }
    delete ctx;
}

} // namespace

WebKitGTKView::WebKitGTKView(GMainWindow& main_window_, Core::System& system_,
                             InputCommon::InputSubsystem* input_subsystem_)
    : QWidget(&main_window_), main_window(main_window_), system(system_),
      input_subsystem(input_subsystem_),
      input_interpreter(std::make_unique<InputInterpreter>(system_)) {
    WebKitUserContentManager* ucm = webkit_user_content_manager_new();

    webkit_user_content_manager_register_script_message_handler(ucm, "nxMessage");
    g_signal_connect(ucm, "script-message-received::nxMessage", G_CALLBACK(OnNxMessage), this);

    webkit_user_content_manager_register_script_message_handler(ucm, "nxControl");
    g_signal_connect(ucm, "script-message-received::nxControl", G_CALLBACK(OnNxControl), this);

    // Inject window_nx + gamepad scripts at document start, all frames -- mirrors qt_web_browser.cpp:62-81.
    // Font/focus scripts are injected at load time via LoadExtractedFonts/FocusFirstLinkElement.
    WebKitUserScript* window_nx_script = webkit_user_script_new(
        WEBKITGTK_NX_SCRIPT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, window_nx_script);
    webkit_user_script_unref(window_nx_script);

    WebKitUserScript* gamepad_script = webkit_user_script_new(
        WEBKITGTK_GAMEPAD_SCRIPT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, gamepad_script);
    webkit_user_script_unref(gamepad_script);

    webview = WEBKIT_WEB_VIEW(g_object_new(WEBKIT_TYPE_WEB_VIEW, "user-content-manager", ucm, nullptr));
    g_signal_connect(webview, "decide-policy", G_CALLBACK(OnDecidePolicy), this);
    g_signal_connect(webview, "close", G_CALLBACK(OnClose), this);

    // TODO: wire persistent storage via webkit_website_data_manager_new -- WebKitGTK
    // requires this at WebContext creation time, not settable after the WebView exists.
    // auto storage_dir = Common::FS::PathToUTF8String(
    //     Common::FS::GetCitronPath(Common::FS::CitronPath::CitronDir) / "webkitgtk");

    setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "1", 1);

    QWidget* view_widget = Embed(&main_window_);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(view_widget);
    setLayout(layout);
}

WebKitGTKView::~WebKitGTKView() {
    SetFinished(true);
    StopInputThread();
    if (gtk_window) {
        gtk_widget_destroy(gtk_window);
    }
}

QWidget* WebKitGTKView::Embed(QWidget* parent) {
    const QString platform = QGuiApplication::platformName();
    if (platform != QStringLiteral("xcb")) {
        FallbackToTopLevelWindow();
        auto* placeholder = new QWidget(parent);
        return placeholder;
    }

    gtk_window = gtk_offscreen_window_new();
    gtk_container_add(GTK_CONTAINER(gtk_window), GTK_WIDGET(webview));
    gtk_widget_realize(gtk_window);
    gtk_widget_show_all(gtk_window);

#if defined(GDK_WINDOWING_X11)
    GdkWindow* gdk_window = gtk_widget_get_window(GTK_WIDGET(webview));
    if (gdk_window && GDK_IS_X11_WINDOW(gdk_window)) {
        Window xid = gdk_x11_window_get_xid(gdk_window);
        QWindow* foreign = QWindow::fromWinId(static_cast<WId>(xid));
        container = QWidget::createWindowContainer(foreign, parent);
        return container;
    }
#endif

    FallbackToTopLevelWindow();
    return new QWidget(parent);
}

void WebKitGTKView::FallbackToTopLevelWindow() {
    gtk_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_container_add(GTK_CONTAINER(gtk_window), GTK_WIDGET(webview));
}

void WebKitGTKView::SetUserAgent(UserAgent user_agent) {
    const QString user_agent_str = [user_agent] {
        switch (user_agent) {
        case UserAgent::WebApplet:
        default:
            return QStringLiteral("WebApplet");
        case UserAgent::ShopN:
            return QStringLiteral("ShopN");
        case UserAgent::LoginApplet:
            return QStringLiteral("LoginApplet");
        case UserAgent::ShareApplet:
            return QStringLiteral("ShareApplet");
        case UserAgent::LobbyApplet:
            return QStringLiteral("LobbyApplet");
        case UserAgent::WifiWebAuthApplet:
            return QStringLiteral("WifiWebAuthApplet");
        }
    }();

    const QString full_ua =
        QStringLiteral("Mozilla/5.0 (Nintendo Switch; %1) AppleWebKit/606.4 "
                       "(KHTML, like Gecko) NF/6.0.1.15.4 NintendoBrowser/5.1.0.20389")
            .arg(user_agent_str);
    WebKitSettings* settings = webkit_web_view_get_settings(webview);
    QByteArray utf8 = full_ua.toUtf8();
    webkit_settings_set_user_agent(settings, utf8.constData());
}

void WebKitGTKView::LoadExtractedFonts() {
    auto fonts_dir_str = Common::FS::PathToUTF8String(
        Common::FS::GetCitronPath(Common::FS::CitronPath::CacheDir) / "fonts/");
    std::replace(fonts_dir_str.begin(), fonts_dir_str.end(), '\\', '/');
    const QString fonts_dir = QString::fromStdString(fonts_dir_str);

    const QString css_source = QString::fromUtf8(WEBKITGTK_NX_FONT_CSS)
                                   .arg(fonts_dir + QStringLiteral("FontStandard.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontChineseSimplified.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontExtendedChineseSimplified.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontChineseTraditional.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontKorean.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontNintendoExtended.ttf"))
                                   .arg(fonts_dir + QStringLiteral("FontNintendoExtended2.ttf"));

    WebKitUserContentManager* ucm = webkit_web_view_get_user_content_manager(webview);

    // WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END approximates Qt's DocumentReady timing.
    QByteArray css_utf8 = css_source.toUtf8();
    WebKitUserScript* css_script = webkit_user_script_new(
        css_utf8.constData(), WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, css_script);
    webkit_user_script_unref(css_script);

    // DOCUMENT_END approximates Qt's Deferred timing; also re-run on every navigation
    // via decide-policy (50 ms debounce), mirroring qt_web_browser.cpp:380-386.
    WebKitUserScript* font_script = webkit_user_script_new(
        WEBKITGTK_LOAD_NX_FONT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, font_script);
    webkit_user_script_unref(font_script);
}

void WebKitGTKView::FocusFirstLinkElement() {
    WebKitUserContentManager* ucm = webkit_web_view_get_user_content_manager(webview);
    WebKitUserScript* script = webkit_user_script_new(
        WEBKITGTK_FOCUS_LINK_ELEMENT_SCRIPT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_END, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, script);
    webkit_user_script_unref(script);
}

void WebKitGTKView::LoadLocalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = true;

    LoadExtractedFonts();
    FocusFirstLinkElement();
    SetUserAgent(UserAgent::WebApplet);
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

    gchar* file_uri = g_filename_to_uri(main_url.c_str(), nullptr, nullptr);
    std::string uri = std::string(file_uri) + additional_args;
    g_free(file_uri);
    webkit_web_view_load_uri(webview, uri.c_str());
    if (gtk_window) {
        gtk_widget_show_all(gtk_window);
    }
}

void WebKitGTKView::LoadExternalWebPage(const std::string& main_url, const std::string& additional_args) {
    is_local = false;

    FocusFirstLinkElement();
    SetUserAgent(UserAgent::WebApplet);
    SetFinished(false);
    SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    SetLastURL("http://localhost/");
    StartInputThread();

    std::string uri = main_url + additional_args;
    webkit_web_view_load_uri(webview, uri.c_str());
    if (gtk_window) {
        gtk_widget_show_all(gtk_window);
    }
}

QString WebKitGTKView::GetCurrentURL() const {
    return requested_url;
}

void WebKitGTKView::EvaluateJavaScript(const QString& script, std::function<void(const QVariant&)> callback) {
    QByteArray utf8 = script.toUtf8();
    if (callback) {
        auto* ctx = new EvalCallbackContext{std::move(callback)};
        webkit_web_view_evaluate_javascript(webview, utf8.constData(), -1, nullptr, nullptr,
                                            nullptr, OnEvaluateJavaScriptFinished, ctx);
    } else {
        webkit_web_view_evaluate_javascript(webview, utf8.constData(), -1, nullptr, nullptr,
                                            nullptr, nullptr, nullptr);
    }
}

void WebKitGTKView::SetPageZoomFactor(qreal factor) {
    webkit_web_view_set_zoom_level(webview, static_cast<gdouble>(factor));
}

void WebKitGTKView::SendKeyEvent(const QString& key, const QString& code, int key_code) {
    const QString script =
        QStringLiteral("(function() { var el = document.activeElement || document.body; "
                       "var opts = { key: '%1', code: '%2', keyCode: %3, which: %3, "
                       "bubbles: true, cancelable: true }; "
                       "el.dispatchEvent(new KeyboardEvent('keydown', opts)); "
                       "el.dispatchEvent(new KeyboardEvent('keyup', opts)); })();")
            .arg(key, code)
            .arg(key_code);
    EvaluateJavaScript(script);
}

void WebKitGTKView::hide() {
    SetFinished(true);
    QWidget::hide();
}

void WebKitGTKView::StartInputThread() {
    if (input_thread_running) {
        return;
    }
    input_thread_running = true;
    input_thread = std::thread(&WebKitGTKView::InputThreadLoop, this);
}

void WebKitGTKView::StopInputThread() {
    if (!input_thread_running) {
        return;
    }
    input_thread_running = false;
    if (input_thread.joinable()) {
        input_thread.join();
    }
}

void WebKitGTKView::InputThreadLoop() {
    // Mirrors qt_web_browser.cpp:222-291's InputThread: 1 s startup delay, then
    // continuous poll with same button mapping and pressed-once vs held logic.
    // Key events are sent via JS eval (SendKeyEvent) rather than Qt postEvent.
    std::this_thread::sleep_for(std::chrono::seconds(1));

    if (is_local) {
        // grabKeyboard()/releaseKeyboard() have no equivalent for a
        // createWindowContainer-embedded foreign window -- not ported.
    }

    while (input_thread_running) {
        input_interpreter->PollInput();

        using Core::HID::NpadButton;
        for (NpadButton button : {NpadButton::A, NpadButton::B, NpadButton::X, NpadButton::Y,
                                  NpadButton::L, NpadButton::R}) {
            if (!input_interpreter->IsButtonPressedOnce(button)) {
                continue;
            }
            int callback_index = -1;
            const char* fallback_key = nullptr;
            int fallback_code = 0;
            switch (button) {
            case NpadButton::A:
                callback_index = 0;
                fallback_key = "a";
                fallback_code = 65;
                break;
            case NpadButton::B:
                callback_index = 1;
                fallback_key = "b";
                fallback_code = 66;
                break;
            case NpadButton::X:
                callback_index = 2;
                fallback_key = "x";
                fallback_code = 88;
                break;
            case NpadButton::Y:
                callback_index = 3;
                fallback_key = "y";
                fallback_code = 89;
                break;
            case NpadButton::L:
                callback_index = 6;
                break;
            case NpadButton::R:
                callback_index = 7;
                break;
            default:
                break;
            }

            // Check if a callback is registered; send fallback key only if not.
            // evaluate_javascript is async, so the decision is gated on the result.
            const QString check_script =
                QStringLiteral("citron_key_callbacks[%1] != null").arg(callback_index);
            EvaluateJavaScript(check_script, [this, callback_index, fallback_key,
                                              fallback_code](const QVariant& has_callback) {
                if (has_callback.toBool()) {
                    EvaluateJavaScript(
                        QStringLiteral("citron_key_callbacks[%1]();").arg(callback_index));
                } else if (fallback_key) {
                    SendKeyEvent(QString::fromUtf8(fallback_key), QString::fromUtf8(fallback_key).toUpper(),
                                fallback_code);
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
                    SendKeyEvent(QString::fromUtf8(dom_key.key), QString::fromUtf8(dom_key.code),
                                dom_key.key_code);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void WebKitGTKView::OnNxMessage(WebKitUserContentManager*, WebKitJavascriptResult* result, gpointer user_data) {
    auto* self = static_cast<WebKitGTKView*>(user_data);
    JSCValue* value = webkit_javascript_result_get_js_value(result);
    char* str_value = jsc_value_to_string(value);
    self->main_window.ForwardWebBrowserInteractiveData(std::string(str_value));
    g_free(str_value);
}

void WebKitGTKView::OnNxControl(WebKitUserContentManager*, WebKitJavascriptResult* result, gpointer user_data) {
    auto* self = static_cast<WebKitGTKView*>(user_data);
    JSCValue* value = webkit_javascript_result_get_js_value(result);
    char* str_value = jsc_value_to_string(value);
    if (strstr(str_value, "\"event\":\"endApplet\"") != nullptr) {
        self->SetFinished(true);
        self->SetExitReason(Service::AM::Frontend::WebExitReason::EndButtonPressed);
    }
    g_free(str_value);
}

int WebKitGTKView::OnDecidePolicy(WebKitWebView*, WebKitPolicyDecision* decision,
                                  int decision_type_raw, gpointer user_data) {
    auto decision_type = static_cast<WebKitPolicyDecisionType>(decision_type_raw);
    auto* self = static_cast<WebKitGTKView*>(user_data);
    if (decision_type != WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION) {
        return FALSE;
    }
    auto* nav_decision = WEBKIT_NAVIGATION_POLICY_DECISION(decision);
    auto* action = webkit_navigation_policy_decision_get_navigation_action(nav_decision);
    auto* request = webkit_navigation_action_get_request(action);
    const char* uri = webkit_uri_request_get_uri(request);

    self->requested_url = QString::fromUtf8(uri);

    if (self->requested_url.contains(QStringLiteral("localhost"))) {
        self->SetFinished(true);
        self->SetExitReason(Service::AM::Frontend::WebExitReason::CallbackURL);
        self->SetLastURL(self->requested_url.toStdString());
    }

    // Re-run load_nx_font on every navigation, mirroring qt_web_browser.cpp:380-386 (50 ms debounce).
    QTimer::singleShot(50, self, [self] { self->EvaluateJavaScript(QString::fromUtf8(WEBKITGTK_LOAD_NX_FONT)); });

    webkit_policy_decision_use(decision);
    return TRUE;
}

void WebKitGTKView::OnClose(WebKitWebView*, gpointer user_data) {
    // Mirrors qt_web_browser.cpp:94-102's windowCloseRequested handler.
    // The "close" signal is already scoped to this specific WebKitWebView,
    // so no URL guard is needed here.
    auto* self = static_cast<WebKitGTKView*>(user_data);
    self->SetFinished(true);
    self->SetExitReason(Service::AM::Frontend::WebExitReason::WindowClosed);
}

#endif // CITRON_USE_WEBKITGTK_WEB_ENGINE
