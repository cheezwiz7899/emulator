// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "citron/applets/webkitgtk_web_browser.h"

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE

#include <QGuiApplication>
#include <QVBoxLayout>
#include <QWindow>

#include <gdk/gdk.h>
#if defined(GDK_WINDOWING_X11)
#include <gdk/gdkx.h>
#endif

#include "citron/applets/webkitgtk_web_browser_scripts.h"
#include "citron/main.h"

namespace {
// Mirrors main.cpp:242's exact eval string shape (button_index substituted).
QString FooterCallbackScript(int button_index) {
    return QStringLiteral("if (citron_key_callbacks[%1] != null) { citron_key_callbacks[%1](); }")
        .arg(button_index)
        .arg(button_index);
}
} // namespace

WebKitGTKView::WebKitGTKView(GMainWindow& main_window_, Core::System& system_,
                             InputCommon::InputSubsystem* input_subsystem_)
    : QWidget(&main_window_), main_window(main_window_), system(system_),
      input_subsystem(input_subsystem_) {
    WebKitUserContentManager* ucm = webkit_user_content_manager_new();

    webkit_user_content_manager_register_script_message_handler(ucm, "nxMessage");
    g_signal_connect(ucm, "script-message-received::nxMessage", G_CALLBACK(OnNxMessage), this);

    webkit_user_content_manager_register_script_message_handler(ucm, "nxControl");
    g_signal_connect(ucm, "script-message-received::nxControl", G_CALLBACK(OnNxControl), this);

    // WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START == QWebEngineScript::DocumentCreation
    // parity, confirmed in this session's earlier research, not re-derived.
    WebKitUserScript* script = webkit_user_script_new(
        WEBKITGTK_NX_SCRIPT, WEBKIT_USER_CONTENT_INJECT_ALL_FRAMES,
        WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, nullptr, nullptr);
    webkit_user_content_manager_add_script(ucm, script);
    webkit_user_script_unref(script);

    webview = WEBKIT_WEB_VIEW(g_object_new(WEBKIT_TYPE_WEB_VIEW, "user-content-manager", ucm, nullptr));
    g_signal_connect(webview, "decide-policy", G_CALLBACK(OnDecidePolicy), this);

    // Same DMABUF-renderer workaround carried from bridge_spike.c's honesty notes --
    // still not validated against real NVIDIA hardware as of this patch, gate
    // unchanged from earlier in this session.
    setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "1", 1);

    QWidget* view_widget = Embed(&main_window_);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(view_widget);
    setLayout(layout);
}

WebKitGTKView::~WebKitGTKView() {
    if (gtk_window) {
        gtk_widget_destroy(gtk_window);
    }
}

QWidget* WebKitGTKView::Embed(QWidget* parent) {
    // See the header's Embed() comment for the full X11-vs-Wayland finding. Runtime
    // check, not a compile-time #ifdef, because the SAME built binary can run under
    // either backend depending on the user's session -- this has to be a runtime
    // decision, unlike the CITRON_USE_WEBKITGTK_WEB_ENGINE build-time backend choice.
    const QString platform = QGuiApplication::platformName();
    if (platform != QStringLiteral("xcb")) {
        // Wayland (or anything else that isn't xcb) -- fall back rather than fail.
        // NOT VALIDATED on a real Wayland session as part of this patch.
        FallbackToTopLevelWindow();
        auto* placeholder = new QWidget(parent); // occupies the layout slot; the
                                                  // real content is the separate
                                                  // top-level gtk_window.
        return placeholder;
    }

    gtk_window = gtk_offscreen_window_new(); // GTK still needs *a* toplevel to own
                                             // the widget hierarchy before its
                                             // native window can be pulled out and
                                             // reparented via createWindowContainer.
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

    // xcb platform reported but couldn't get an X11 GdkWindow -- shouldn't happen
    // given the platformName() check above, but fail safe rather than crash.
    FallbackToTopLevelWindow();
    return new QWidget(parent);
}

void WebKitGTKView::FallbackToTopLevelWindow() {
    // This is deliberately the same shape as bridge_spike.c's standalone GtkWindow
    // -- not a coincidence, see header comment. Floating window instead of
    // replacing the game view seamlessly; ships without solving embedding on
    // Wayland.
    gtk_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_container_add(GTK_CONTAINER(gtk_window), GTK_WIDGET(webview));
}

void WebKitGTKView::LoadLocalWebPage(const std::string& main_url, const std::string& additional_args) {
    // Mirrors QtNXWebEngineView::LoadLocalWebPage (qt_web_browser.cpp:110-122):
    // QUrl::fromLocalFile(...).toString() + additional_args.
    gchar* file_uri = g_filename_to_uri(main_url.c_str(), nullptr, nullptr);
    std::string uri = std::string(file_uri) + additional_args;
    g_free(file_uri);
    webkit_web_view_load_uri(webview, uri.c_str());
    if (gtk_window) {
        gtk_widget_show_all(gtk_window);
    }
}

void WebKitGTKView::LoadExternalWebPage(const std::string& main_url, const std::string& additional_args) {
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
    // Fire-and-forget when no callback given, matching how main.cpp's 3 call sites
    // for this (983, 1001, 1079) are used -- none of them currently consume the
    // QVariant result, they use the callback purely for completion timing. A
    // real callback bridge (webkit_web_view_evaluate_javascript_finish ->
    // JSCValue -> QVariant) isn't wired here since nothing in the audited call
    // sites needs it -- flagging as sketched, not silently dropped.
    QByteArray utf8 = script.toUtf8();
    webkit_web_view_evaluate_javascript(webview, utf8.constData(), -1, nullptr, nullptr,
                                        nullptr, nullptr, nullptr);
    if (callback) {
        callback(QVariant());
    }
}

void WebKitGTKView::SetPageZoomFactor(qreal factor) {
    webkit_web_view_set_zoom_level(webview, static_cast<gdouble>(factor));
}

void WebKitGTKView::hide() {
    SetFinished(true);
    QWidget::hide();
}

void WebKitGTKView::OnNxMessage(WebKitUserContentManager*, WebKitJavascriptResult* result, gpointer user_data) {
    auto* self = static_cast<WebKitGTKView*>(user_data);
    JSCValue* value = webkit_javascript_result_get_js_value(result);
    char* str_value = jsc_value_to_string(value);
    // Push-based equivalent of main.cpp:1007's WebBrowserInteractiveDataReceived
    // emit -- fires the moment the message arrives instead of being polled.
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

gboolean WebKitGTKView::OnDecidePolicy(WebKitWebView*, WebKitPolicyDecision* decision,
                                       WebKitPolicyDecisionType decision_type, gpointer user_data) {
    auto* self = static_cast<WebKitGTKView*>(user_data);
    if (decision_type != WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION) {
        return FALSE;
    }
    auto* nav_decision = WEBKIT_NAVIGATION_POLICY_DECISION(decision);
    auto* action = webkit_navigation_policy_decision_get_navigation_action(nav_decision);
    auto* request = webkit_navigation_action_get_request(action);
    const char* uri = webkit_uri_request_get_uri(request);

    self->requested_url = QString::fromUtf8(uri);

    // Mirrors main.cpp:1012-1019's GetCurrentURL().contains("localhost") check --
    // moved earlier (into the event itself, at decide-policy time) rather than
    // polled, same poll-elimination reasoning as legs 1/3.
    if (self->requested_url.contains(QStringLiteral("localhost"))) {
        self->SetFinished(true);
        self->SetExitReason(Service::AM::Frontend::WebExitReason::CallbackURL);
        self->SetLastURL(self->requested_url.toStdString());
    }

    webkit_policy_decision_use(decision); // observe-only, matches UrlRequestInterceptor
    return TRUE;
}

#endif // CITRON_USE_WEBKITGTK_WEB_ENGINE
