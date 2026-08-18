// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// All 5 scripts, same set/provenance as webkitgtk_web_browser_scripts.h. See
// that file's header for why this exists (missed on first pass, found on full
// re-read of qt_web_browser.cpp).

#pragma once

constexpr wchar_t WEBVIEW2_NX_FONT_CSS[] = LR"(
(function() {
    css = document.createElement('style');
    css.type = 'text/css';
    css.id = 'nx_font';
    css.innerText = `
/* FontStandard */
@font-face {
    font-family: 'FontStandard';
    src: url('%1') format('truetype');
}

/* FontChineseSimplified */
@font-face {
    font-family: 'FontChineseSimplified';
    src: url('%2') format('truetype');
}

/* FontExtendedChineseSimplified */
@font-face {
    font-family: 'FontExtendedChineseSimplified';
    src: url('%3') format('truetype');
}

/* FontChineseTraditional */
@font-face {
    font-family: 'FontChineseTraditional';
    src: url('%4') format('truetype');
}

/* FontKorean */
@font-face {
    font-family: 'FontKorean';
    src: url('%5') format('truetype');
}

/* FontNintendoExtended */
@font-face {
    font-family: 'NintendoExt003';
    src: url('%6') format('truetype');
}

/* FontNintendoExtended2 */
@font-face {
    font-family: 'NintendoExt003';
    src: url('%7') format('truetype');
}
`;

    document.head.appendChild(css);
})();
)";

constexpr wchar_t WEBVIEW2_LOAD_NX_FONT[] = LR"(
(function() {
    var elements = document.querySelectorAll("*");

    for (var i = 0; i < elements.length; i++) {
        var style = window.getComputedStyle(elements[i], null);
        if (style.fontFamily.includes("Arial") || style.fontFamily.includes("Calibri") ||
            style.fontFamily.includes("Century") || style.fontFamily.includes("Times New Roman")) {
            elements[i].style.fontFamily = "FontStandard, FontChineseSimplified, FontExtendedChineseSimplified, FontChineseTraditional, FontKorean, NintendoExt003";
        } else {
            elements[i].style.fontFamily = style.fontFamily + ", FontStandard, FontChineseSimplified, FontExtendedChineseSimplified, FontChineseTraditional, FontKorean, NintendoExt003";
        }
    }
})();
)";

constexpr wchar_t WEBVIEW2_FOCUS_LINK_ELEMENT_SCRIPT[] = LR"(
if (document.getElementsByTagName("a").length > 0) {
    document.getElementsByTagName("a")[0].focus();
}
)";

constexpr wchar_t WEBVIEW2_GAMEPAD_SCRIPT[] = LR"(
window.addEventListener("gamepadconnected", function(e) {
    console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
        e.gamepad.index, e.gamepad.id, e.gamepad.buttons.length, e.gamepad.axes.length);
});

window.addEventListener("gamepaddisconnected", function(e) {
    console.log("Gamepad disconnected from index %d: %s", e.gamepad.index, e.gamepad.id);
});
)";

constexpr wchar_t WEBVIEW2_NX_SCRIPT[] = LR"(
// Ported from src/citron/applets/qt_web_browser_scripts.h:92-204 (WINDOW_NX_SCRIPT),
// same base as nx_shim.js (WebKitGTK port). WebView2 target instead.
//
// UNVERIFIED BY EXECUTION. No WebView2 runtime available in this sandbox (no Windows).
// API surface below confirmed via Microsoft Learn / official WebView2 docs (search,
// this session) -- not from training-data recall, not from a REPL. Treat as a strong
// draft, not proven, unlike nx_shim.js which was actually run.
//
// Mechanism changes vs the original (2 total, same reasoning as the WebKitGTK port --
// eliminate main.cpp's poll loop, push instead):
//
//   1. sendMessage(message): window.chrome.webview.postMessage(message) directly.
//      Confirmed: postMessage takes "any object supported by JSON conversion" and is
//      posted async to the host (Microsoft Learn, ICoreWebView2 reference). Since the
//      existing shim already only ever passes strings (page does its own
//      JSON.stringify() first), this is a drop-in call, no wrapping needed.
//
//   2. endApplet(): CANNOT use a second named handler the way WebKitGTK's nxControl
//      did -- WebView2 has exactly one JS->native channel (window.chrome.webview /
//      WebMessageReceived), not per-name registration. So this is envelope-tagged
//      instead: posts a small object with a __citron_control key. Native side
//      distinguishes "is this the control envelope, or an opaque page payload" by
//      checking for that key before treating it as a regular sendMessage forward.
//      This is a real, necessary design difference from the WebKitGTK port, not an
//      oversight -- flagged in webview2_bridge.cpp's on_web_message_received too.
//
// Footer key-callback dispatch (citron_key_callbacks) UNCHANGED, same as the
// WebKitGTK port -- still native-polls-input -> eval-into-page, platform-agnostic
// concern, not touched here either.

var citron_key_callbacks = [];

(function() {
    class WindowNX {
        constructor() {
            citron_key_callbacks[1] = function() { window.history.back(); };
            citron_key_callbacks[2] = function() { window.nx.endApplet(); };
        }

        addEventListener(type, listener, options) {
            console.log("nx.addEventListener called, type=%s", type);

            window.addEventListener(type, listener, options);
        }

        endApplet() {
            console.log("nx.endApplet called");

            window.chrome.webview.postMessage({ __citron_control: "endApplet" });
        }

        playSystemSe(system_se) {
            console.log("nx.playSystemSe is not implemented, system_se=%s", system_se);
        }

        sendMessage(message) {
            console.log("nx.sendMessage called, message=%s", message);

            window.chrome.webview.postMessage(message);
        }

        setCursorScrollSpeed(scroll_speed) {
            console.log("nx.setCursorScrollSpeed is not implemented, scroll_speed=%d", scroll_speed);
        }
    }

    class WindowNXFooter {
        setAssign(key, label, func, option) {
            console.log("nx.footer.setAssign called, key=%s", key);

            switch (key) {
                case "A":
                    citron_key_callbacks[0] = func;
                    break;
                case "B":
                    citron_key_callbacks[1] = func;
                    break;
                case "X":
                    citron_key_callbacks[2] = func;
                    break;
                case "Y":
                    citron_key_callbacks[3] = func;
                    break;
                case "L":
                    citron_key_callbacks[6] = func;
                    break;
                case "R":
                    citron_key_callbacks[7] = func;
                    break;
            }
        }

        setFixed(kind) {
            console.log("nx.footer.setFixed is not implemented, kind=%s", kind);
        }

        unsetAssign(key) {
            console.log("nx.footer.unsetAssign called, key=%s", key);

            switch (key) {
                case "A":
                    citron_key_callbacks[0] = function() {};
                    break;
                case "B":
                    citron_key_callbacks[1] = function() {};
                    break;
                case "X":
                    citron_key_callbacks[2] = function() {};
                    break;
                case "Y":
                    citron_key_callbacks[3] = function() {};
                    break;
                case "L":
                    citron_key_callbacks[6] = function() {};
                    break;
                case "R":
                    citron_key_callbacks[7] = function() {};
                    break;
            }
        }
    }

    class WindowNXPlayReport {
        incrementCounter(counter_id) {
            console.log("nx.playReport.incrementCounter is not implemented, counter_id=%d", counter_id);
        }

        setCounterSetIdentifier(counter_id) {
            console.log("nx.playReport.setCounterSetIdentifier is not implemented, counter_id=%d", counter_id);
        }
    }

    window.nx = new WindowNX();
    window.nx.footer = new WindowNXFooter();
    window.nx.playReport = new WindowNXPlayReport();
})();
)";
