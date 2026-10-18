// SPDX-FileCopyrightText: 2026 BorbEMU
// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.thor

import android.app.Activity
import android.app.ActivityOptions
import android.content.Context
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Bundle
import android.view.Display

/**
 * The second built-in screen of a dual-screen handheld.
 *
 * Android lists a television, a cast target and a screen recording the same way it lists the
 * AYN Thor's bottom screen: as one more display carrying the presentation flag. Nothing in the
 * public API tells them apart on the Thor (both of its panels report an unknown connection
 * type), so the handhelds that have one are named here. On every other device [find] returns
 * null, no panel is shown, the touch controls stay where they are and a game stays on the
 * screen Android put it on.
 */
object SecondScreen {
    private val dualScreenHandheld: Boolean by lazy {
        Build.MANUFACTURER.equals("AYN", ignoreCase = true) &&
            Build.MODEL.contains("Thor", ignoreCase = true)
    }

    /** The second built-in screen, or null when the device has none or it is switched off. */
    fun find(context: Context): Display? {
        if (!dualScreenHandheld) {
            return null
        }
        val manager = context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
        return manager.displays.firstOrNull {
            it.displayId != Display.DEFAULT_DISPLAY && it.state != Display.STATE_OFF
        }
    }

    /** True when [activity] is showing on the second screen and not on the main one. */
    fun isOn(activity: Activity): Boolean {
        val second = find(activity) ?: return false
        return activity.display?.displayId == second.displayId
    }

    /** Launch options that put an activity on the main screen. */
    fun mainScreenOptions(): Bundle =
        ActivityOptions.makeBasic().setLaunchDisplayId(Display.DEFAULT_DISPLAY).toBundle()
}
