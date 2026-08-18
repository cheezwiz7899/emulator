// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// All 5 scripts qt_web_browser_scripts.h defines (NX_FONT_CSS, LOAD_NX_FONT,
// FOCUS_LINK_ELEMENT_SCRIPT, GAMEPAD_SCRIPT, WINDOW_NX_SCRIPT), byte-identical
// except WINDOW_NX_SCRIPT's 2 documented mechanism changes (see that block below).
// Missed all but WINDOW_NX_SCRIPT in the first pass of this patch -- found on a
// full re-read of qt_web_browser.cpp before calling anything ready to ship.

#pragma once

constexpr char WEBKITGTK_NX_FONT_CSS[] = R"(
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

constexpr char WEBKITGTK_LOAD_NX_FONT[] = R"(
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

constexpr char WEBKITGTK_FOCUS_LINK_ELEMENT_SCRIPT[] = R"(
if (document.getElementsByTagName("a").length > 0) {
    document.getElementsByTagName("a")[0].focus();
}
)";

constexpr char WEBKITGTK_GAMEPAD_SCRIPT[] = R"(
window.addEventListener("gamepadconnected", function(e) {
    console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
        e.gamepad.index, e.gamepad.id, e.gamepad.buttons.length, e.gamepad.axes.length);
});

window.addEventListener("gamepaddisconnected", function(e) {
    console.log("Gamepad disconnected from index %d: %s", e.gamepad.index, e.gamepad.id);
});
)";

constexpr char WEBKITGTK_NX_SCRIPT[] = R"(
// Ported from src/citron/applets/qt_web_browser_scripts.h:92-204 (WINDOW_NX_SCRIPT).
// Everything is byte-identical to that file EXCEPT two mechanism changes, both
// called out in handoff v2 Part 2 ("WebKitGTK's message-handler mechanism is
// arguably cleaner than the current sendMessage-outgoing-queue-poll design
// already in qt_web_browser.cpp -- worth reconsidering that part of the bridge
// design during the port rather than transplanting it as-is"):
//
//   1. sendMessage(message): originally pushed onto a `citron_outgoing_messages`
//      array that main.cpp then drained by round-tripping a runJavaScript eval
//      every ~1ms (main.cpp:1001-1003). Replaced with a direct
//      window.webkit.messageHandlers.nxMessage.postMessage(message) push --
//      no native-side poll loop needed at all, WebKitGTK delivers it as a
//      script-message-received signal.
//
//   2. endApplet(): originally set a module-level `end_applet` boolean that
//      main.cpp polled via a second runJavaScript eval every loop iteration
//      (main.cpp:983-991, "end_applet;"). Replaced with a
//      nxControl.postMessage(...) push, same reasoning as (1).
//
// `end_applet` and `citron_outgoing_messages` vars are dropped since nothing
// reads them anymore under the push design. citron_key_callbacks is UNCHANGED --
// footer-button dispatch is still native-polls-input -> eval-into-page, which
// is inherent to how gamepad input has to be polled regardless of backend, not
// the same "smell" as the outgoing-message array. Not touched, not in scope.
//
// Everything else (WindowNXFooter, WindowNXPlayReport, all stub methods,
// console.log wording) is untouched on purpose -- minimum diff from
// proven-correct source.

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

            window.webkit.messageHandlers.nxControl.postMessage(JSON.stringify({event: "endApplet"}));
        }

        playSystemSe(system_se) {
            console.log("nx.playSystemSe is not implemented, system_se=%s", system_se);
        }

        sendMessage(message) {
            console.log("nx.sendMessage called, message=%s", message);

            window.webkit.messageHandlers.nxMessage.postMessage(message);
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
