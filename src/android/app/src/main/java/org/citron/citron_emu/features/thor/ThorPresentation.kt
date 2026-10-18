package org.citron.citron_emu.features.thor

import android.app.Presentation
import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Display
import android.view.View
import android.widget.TextView
import org.citron.citron_emu.NativeLibrary
import org.citron.citron_emu.R

import org.citron.citron_emu.features.settings.model.BooleanSetting
import org.citron.citron_emu.features.settings.model.IntSetting

class ThorPresentation(context: Context, display: Display) : Presentation(context, display) {
    private val handler = Handler(Looper.getMainLooper())
    private val updateRunnable = object : Runnable {
        override fun run() {
            updateStats()
            handler.postDelayed(this, 500)
        }
    }

    private lateinit var fpsText: TextView
    private lateinit var speedText: TextView
    private lateinit var gpuText: TextView
    private lateinit var shadersText: TextView
    private lateinit var resText: TextView
    private lateinit var dockText: TextView
    private lateinit var accText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.thor_presentation)

        fpsText = findViewById(R.id.thor_fps_text)
        speedText = findViewById(R.id.thor_speed_text)
        gpuText = findViewById(R.id.thor_gpu_text)
        shadersText = findViewById(R.id.thor_shaders_text)
        resText = findViewById(R.id.thor_setting_res)
        dockText = findViewById(R.id.thor_setting_dock)
        accText = findViewById(R.id.thor_setting_acc)

        // Initial GPU Driver check
        try {
            gpuText.text = "GPU: " + NativeLibrary.getGpuDriver()
        } catch (e: Exception) {
            gpuText.text = "GPU: Detecting..."
        }
    }

    override fun onStart() {
        super.onStart()
        handler.post(updateRunnable)
    }

    override fun onStop() {
        super.onStop()
        handler.removeCallbacks(updateRunnable)
    }

    private fun updateStats() {
        // Update Settings
        val resVal = IntSetting.RENDERER_RESOLUTION.getInt()
        // The numbers are the ones in rendererResolutionValues (arrays.xml).
        val resString = when(resVal) {
            0 -> "0.25X"
            1 -> "0.5X"
            2 -> "0.75X"
            3 -> "1X"
            12 -> "1.25X"
            4 -> "1.5X"
            13 -> "1.75X"
            5 -> "2X"
            6 -> "3X"
            7 -> "4X"
            8 -> "5X"
            9 -> "6X"
            10 -> "7X"
            11 -> "8X"
            else -> "?"
        }
        resText.text = "Res: $resString"

        val docked = BooleanSetting.USE_DOCKED_MODE.getBoolean()
        dockText.text = if (docked) "Docked" else "Handheld"
        dockText.setTextColor(if (docked) 0xFF55AAFF.toInt() else 0xFFAAFFAA.toInt())

        val accVal = IntSetting.CPU_ACCURACY.getInt()
        val accString = when(accVal) {
            0 -> "Auto"
            1 -> "Accurate"
            2 -> "Unsafe"
            3 -> "Paranoid"
            else -> "?"
        }
        accText.text = "Acc: $accString"

        if (!NativeLibrary.isRunning()) return

        // onCreate can run before the renderer exists. When it does, this
        // line read "GPU:" with nothing after it for the whole session.
        if (gpuText.text.length <= "GPU: ".length) {
            gpuText.text = "GPU: " + NativeLibrary.getGpuDriver()
        }

        val stats = NativeLibrary.getPerfStats() // [FPS, GameFPS, FrameTime, Speed]
        if (stats.size >= 4) {
            val fps = stats[0]
            val speed = stats[3] * 100

            fpsText.text = String.format("FPS: %.1f", fps)
            speedText.text = String.format("Speed: %.0f%%", speed)
        }

        val shaders = NativeLibrary.getShadersBuilding()
        if (shaders > 0) {
            shadersText.text = "Shaders Building: $shaders"
            shadersText.visibility = View.VISIBLE
        } else {
            shadersText.visibility = View.INVISIBLE
        }
    }
}
