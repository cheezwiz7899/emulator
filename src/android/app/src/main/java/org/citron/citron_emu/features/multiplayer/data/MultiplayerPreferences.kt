// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.data

import androidx.preference.PreferenceManager
import org.citron.citron_emu.CitronApplication
import org.citron.citron_emu.features.multiplayer.model.DirectConnectParams
import org.citron.citron_emu.features.multiplayer.model.HostRoomParams
import org.citron.citron_emu.features.multiplayer.model.MultiplayerValidation

object MultiplayerPreferences {
    private val preferences
        get() = PreferenceManager.getDefaultSharedPreferences(CitronApplication.appContext)

    data class DirectConnectForm(val host: String, val port: Int, val nickname: String)

    data class HostRoomForm(
        val roomName: String,
        val nickname: String,
        val description: String,
        val port: Int,
        val maxPlayers: Int
    )

    fun loadDirectConnect(): DirectConnectForm = DirectConnectForm(
        host = preferences.getString(KEY_DIRECT_HOST, "").orEmpty(),
        port = loadSharedPort(),
        nickname = loadSharedNickname()
    )

    fun loadHostRoom(): HostRoomForm = HostRoomForm(
        roomName = preferences.getString(KEY_ROOM_NAME, "").orEmpty(),
        nickname = loadSharedNickname(),
        description = preferences.getString(KEY_DESCRIPTION, "").orEmpty(),
        port = loadSharedPort(),
        maxPlayers = preferences.getInt(KEY_MAX_PLAYERS, DEFAULT_MAX_PLAYERS)
    )

    fun saveDirectConnect(params: DirectConnectParams) {
        saveDirectConnectForm(DirectConnectForm(params.host, params.port, params.nickname))
    }

    fun saveHostRoom(params: HostRoomParams) {
        saveHostRoomForm(
            HostRoomForm(
                params.roomName,
                params.nickname,
                params.description,
                params.port,
                params.maxPlayers
            )
        )
    }

    fun saveDirectConnectForm(form: DirectConnectForm) {
        preferences.edit()
            .putString(KEY_DIRECT_HOST, form.host)
            .putString(KEY_NICKNAME, form.nickname)
            .also { editor ->
                if (form.port in 1..MultiplayerValidation.MAX_PORT) {
                    editor.putInt(KEY_PORT, form.port)
                }
            }
            .apply()
    }

    fun saveHostRoomForm(form: HostRoomForm) {
        preferences.edit()
            .putString(KEY_ROOM_NAME, form.roomName)
            .putString(KEY_NICKNAME, form.nickname)
            .putString(KEY_DESCRIPTION, form.description)
            .also { editor ->
                if (form.port in 1..MultiplayerValidation.MAX_PORT) {
                    editor.putInt(KEY_PORT, form.port)
                }
                if (form.maxPlayers in MultiplayerValidation.MIN_PLAYERS..MultiplayerValidation.MAX_PLAYERS) {
                    editor.putInt(KEY_MAX_PLAYERS, form.maxPlayers)
                }
            }
            .apply()
    }

    private fun loadSharedNickname(): String = preferences.getString(KEY_NICKNAME, null)
        ?: preferences.getString(LEGACY_HOST_NICKNAME, null)
        ?: preferences.getString(LEGACY_DIRECT_NICKNAME, "").orEmpty()

    private fun loadSharedPort(): Int = when {
        preferences.contains(KEY_PORT) -> preferences.getInt(KEY_PORT, MultiplayerValidation.DEFAULT_PORT)
        preferences.contains(LEGACY_HOST_PORT) -> preferences.getInt(LEGACY_HOST_PORT, MultiplayerValidation.DEFAULT_PORT)
        else -> preferences.getInt(LEGACY_DIRECT_PORT, MultiplayerValidation.DEFAULT_PORT)
    }

    private const val KEY_DIRECT_HOST = "DirectConnectHost"
    private const val KEY_ROOM_NAME = "HostRoomName"
    private const val KEY_DESCRIPTION = "HostRoomDescription"
    private const val KEY_MAX_PLAYERS = "HostRoomMaxPlayers"
    private const val KEY_NICKNAME = "MultiplayerNickname"
    private const val KEY_PORT = "MultiplayerPort"
    private const val LEGACY_DIRECT_NICKNAME = "DirectConnectNickname"
    private const val LEGACY_HOST_NICKNAME = "HostRoomNickname"
    private const val LEGACY_DIRECT_PORT = "DirectConnectPort"
    private const val LEGACY_HOST_PORT = "HostRoomPort"
    private const val DEFAULT_MAX_PLAYERS = 8
}
