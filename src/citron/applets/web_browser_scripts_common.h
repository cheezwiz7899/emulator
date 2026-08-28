// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Ported from qt_web_browser_scripts.h. Shared verbatim across both native
// backends (WebKitGTK, WebView2) -- unlike WINDOW_NX_SCRIPT, none of these 4
// need any per-backend mechanism change, so they were byte-identical copies
// (differing only in char vs wchar_t storage type) until consolidated here.
// WebView2 widens these at the one call site that needs wchar_t
// (Utf8ToWide in webview2_web_browser.cpp) rather than storing a second copy.

#pragma once

constexpr char WEB_BROWSER_NX_FONT_CSS[] = R"(
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

constexpr char WEB_BROWSER_LOAD_NX_FONT[] = R"(
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

constexpr char WEB_BROWSER_FOCUS_LINK_ELEMENT_SCRIPT[] = R"(
if (document.getElementsByTagName("a").length > 0) {
    document.getElementsByTagName("a")[0].focus();
}
)";

constexpr char WEB_BROWSER_GAMEPAD_SCRIPT[] = R"(
window.addEventListener("gamepadconnected", function(e) {
    console.log("Gamepad connected at index %d: %s. %d buttons, %d axes.",
        e.gamepad.index, e.gamepad.id, e.gamepad.buttons.length, e.gamepad.axes.length);
});

window.addEventListener("gamepaddisconnected", function(e) {
    console.log("Gamepad disconnected from index %d: %s", e.gamepad.index, e.gamepad.id);
});
)";
