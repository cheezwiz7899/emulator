// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// WebKitGTK backend for the web browser applet. Alternative to QtNXWebEngineView
// (qt_web_browser.h).
//
// v2 of this file. v1 covered the bridge mechanism only (script injection, message
// handlers, nav interception, eval) and missed real functionality that lives in
// qt_web_browser.cpp outside main.cpp's call sites: font loading/injection, the
// input thread that actually drives footer-callback/D-pad navigation, user agent,
// persistent storage, focus-first-link, and the window.close() exit path. Found on
// a full re-read of qt_web_browser.cpp, not assumed complete after the first pass.
//
// UNCOMPILED. No Qt6 + WebKitGTK combined toolchain in the sandbox this was written
// in -- see webkitgtk_web_browser.cpp's own header for what WAS compile-checked
// this round.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE
#include <QWidget>

// Forward declarations only -- NOT #include <gtk/gtk.h> / <webkit2/webkit2.h>
// here. This header is now transitively included by main.cpp (via the
// ActiveWebEngineView alias machinery), and GTK/GLib headers define struct
// fields literally named "signals"/"slots" that collide with Qt's macros of the
// same name -- main.cpp uses bare emit/signals:/slots: throughout, so it can
// never see the real GTK headers. Only webkitgtk_web_browser.cpp (which needs
// the real definitions to actually call these APIs) includes the real headers,
// scoped there with -DQT_NO_KEYWORDS (src/citron/CMakeLists.txt). Found this the
// hard way via a real compile-check this session, not anticipated up front.
//
// GObject/GTK opaque types are typedef'd structs (typedef struct _X X;) in their
// real headers -- matching the same underlying tag name here is
// forward-declaration-compatible with the real definition when the .cpp sees
// both, not a conflicting redeclaration.
extern "C" {
typedef struct _GtkWidget GtkWidget;
typedef struct _WebKitWebView WebKitWebView;
typedef struct _WebKitUserContentManager WebKitUserContentManager;
typedef struct _WebKitJavascriptResult WebKitJavascriptResult;
typedef struct _WebKitPolicyDecision WebKitPolicyDecision;
typedef void* gpointer; // matches glib's own typedef; avoids pulling glib.h in
                        // just for this one alias
}
#endif

#include "core/frontend/applets/web_browser.h"

class InputInterpreter;

class GMainWindow;

namespace Core {
class System;
}

namespace InputCommon {
class InputSubsystem;
}

namespace Core::HID {
enum class NpadButton : u64;
}

#ifdef CITRON_USE_WEBKITGTK_WEB_ENGINE

class WebKitGTKView : public QWidget {
public:
    // Mirrors QtNXWebEngineView::UserAgent (qt_web_browser.h) exactly -- same enum,
    // same values, same purpose (Nintendo*Browser UA string suffix).
    enum class UserAgent {
        WebApplet,
        ShopN,
        LoginApplet,
        ShareApplet,
        LobbyApplet,
        WifiWebAuthApplet,
    };

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

    [[nodiscard]] QString GetCurrentURL() const;

    void EvaluateJavaScript(const QString& script,
                            std::function<void(const QVariant&)> callback = {});
    void SetPageZoomFactor(qreal factor);

    void hide();

private:
    void SetUserAgent(UserAgent user_agent);
    void LoadExtractedFonts();   // local pages only, mirrors LoadExtractedFonts()
    void FocusFirstLinkElement(); // both local and external, mirrors same-named method

    void StartInputThread();
    void StopInputThread();
    void InputThreadLoop();
    // Sends a synthesized DOM KeyboardEvent via eval instead of Qt's
    // QCoreApplication::postEvent(focusProxy(), ...) -- deliberate deviation, not
    // an oversight. postEvent-to-focusProxy relies on the target being a real
    // Qt-integrated widget; a createWindowContainer-embedded foreign window's key
    // events are handled by GTK's own native event loop, not Qt's, so postEvent
    // would very likely never reach the page at all. Dispatching a synthetic
    // KeyboardEvent through the same eval path already proven for everything else
    // sidesteps that entirely and works identically regardless of embed vs
    // fallback-toplevel path (Embed()/FallbackToTopLevelWindow()).
    void SendKeyEvent(const QString& key, const QString& code, int key_code);

    QWidget* Embed(QWidget* parent);
    void FallbackToTopLevelWindow();

    static void OnNxMessage(WebKitUserContentManager*, WebKitJavascriptResult*, gpointer);
    static void OnNxControl(WebKitUserContentManager*, WebKitJavascriptResult*, gpointer);
    // decision_type is WebKitPolicyDecisionType (a plain C enum, not forward-
    // declarable without its full enumerator list) -- erased to int here, the
    // real typed signature is registered with g_signal_connect from the .cpp,
    // where the full header IS visible. GTK dispatches by the signal name
    // string, not by the C function pointer's declared type matching some
    // caller-visible declaration, so this erasure is safe: G_CALLBACK() is a
    // blind reinterpret_cast in GTK's own macro, not something this header's
    // declared signature needs to match exactly for the connection to work.
    static int OnDecidePolicy(WebKitWebView*, WebKitPolicyDecision*, int, gpointer);
    static void OnClose(WebKitWebView*, gpointer);

    GMainWindow& main_window;
    GtkWidget* gtk_window = nullptr; // only set if FallbackToTopLevelWindow() path taken
    WebKitWebView* webview = nullptr;
    QWidget* container = nullptr; // createWindowContainer() result, X11 path only

    Core::System& system;
    InputCommon::InputSubsystem* input_subsystem;
    std::unique_ptr<InputInterpreter> input_interpreter;
    std::thread input_thread;
    std::atomic<bool> input_thread_running{};

    bool is_local = false;
    std::atomic<bool> finished{false};
    Service::AM::Frontend::WebExitReason exit_reason{};
    std::string last_url;
    mutable QString requested_url; // set by OnDecidePolicy, read by GetCurrentURL
};

#endif // CITRON_USE_WEBKITGTK_WEB_ENGINE
