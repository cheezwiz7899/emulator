// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.data

import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.citron.citron_emu.NativeLibrary
import org.citron.citron_emu.features.multiplayer.model.DirectConnectParams
import org.citron.citron_emu.features.multiplayer.model.HostRoomParams
import org.citron.citron_emu.features.multiplayer.model.MultiplayerError
import org.citron.citron_emu.features.multiplayer.model.MultiplayerSnapshot
import org.citron.citron_emu.features.multiplayer.model.MultiplayerValidation
import org.citron.citron_emu.features.multiplayer.model.RoomConnectionState
import org.citron.citron_emu.features.multiplayer.model.RoomContent
import org.citron.citron_emu.features.multiplayer.model.RoomEvent
import org.citron.citron_emu.features.multiplayer.model.RoomInfo
import org.citron.citron_emu.features.multiplayer.model.RoomMember
import org.citron.citron_emu.features.multiplayer.model.StartRoomResult
import org.citron.citron_emu.features.settings.model.BooleanSetting
import org.citron.citron_emu.utils.NativeConfig

object MultiplayerRepository {
    private val roomMutex = Mutex()

    fun getSnapshot(): MultiplayerSnapshot {
        val state = RoomConnectionState.fromNative(NativeLibrary.getRoomConnectionState())
        val error = if (state == RoomConnectionState.IDLE) {
            MultiplayerError.fromNative(NativeLibrary.getRoomLastError())
        } else {
            null
        }
        return MultiplayerSnapshot(
            connectionState = state,
            isHosting = NativeLibrary.isHostingRoom(),
            isEmulationRunning = NativeLibrary.isRunning(),
            lastError = error
        )
    }

    fun isAirplaneModeEnabled(): Boolean = BooleanSetting.AIRPLANE_MODE.getBoolean()

    fun setAirplaneModeEnabled(enabled: Boolean) {
        BooleanSetting.AIRPLANE_MODE.setBoolean(enabled)
        NativeConfig.saveGlobalConfig()
    }

    suspend fun connect(params: DirectConnectParams): StartRoomResult = roomMutex.withLock {
        val current = getSnapshot()
        if (!current.canStartConnection) {
            return@withLock StartRoomResult(false, MultiplayerError.MEMBER_BUSY)
        }
        val airplaneModeDisabled = disableAirplaneMode()
        MultiplayerPreferences.saveDirectConnect(params)
        val started = NativeLibrary.connectToRoom(
            params.nickname,
            params.host,
            params.port,
            params.password
        )
        StartRoomResult(
            started = started,
            error = if (started) null else awaitError(),
            airplaneModeDisabled = airplaneModeDisabled,
            snapshot = getSnapshot()
        )
    }

    suspend fun host(params: HostRoomParams): StartRoomResult = roomMutex.withLock {
        val current = getSnapshot()
        if (!current.canStartConnection) {
            return@withLock StartRoomResult(false, MultiplayerError.MEMBER_BUSY)
        }
        val airplaneModeDisabled = disableAirplaneMode()
        MultiplayerPreferences.saveHostRoom(params)
        val started = NativeLibrary.hostRoom(
            params.nickname,
            params.roomName,
            params.description,
            params.port,
            params.password,
            params.maxPlayers
        )
        StartRoomResult(
            started = started,
            error = if (started) null else awaitError(),
            airplaneModeDisabled = airplaneModeDisabled,
            snapshot = getSnapshot()
        )
    }

    suspend fun leaveOrClose(hosting: Boolean) = roomMutex.withLock {
        if (hosting) NativeLibrary.closeRoom() else NativeLibrary.leaveRoom()
    }

    fun sendChatMessage(message: String): Boolean =
        MultiplayerValidation.isValidChatMessage(message) &&
            NativeLibrary.sendRoomChatMessage(message)

    fun getRoomContent(): RoomContent = RoomContent(
        info = parseRoomInfo(NativeLibrary.getRoomInfo()),
        members = parseMembers(NativeLibrary.getRoomMembers()),
        events = parseEvents(NativeLibrary.drainRoomEvents())
    )

    private fun disableAirplaneMode(): Boolean {
        if (!isAirplaneModeEnabled()) return false
        setAirplaneModeEnabled(false)
        return true
    }

    private suspend fun awaitError(): MultiplayerError {
        repeat(ERROR_POLL_ATTEMPTS) {
            MultiplayerError.fromNative(NativeLibrary.getRoomLastError())?.let { return it }
            delay(ERROR_POLL_DELAY_MS)
        }
        return MultiplayerError.UNKNOWN
    }

    private fun parseRoomInfo(values: Array<String>): RoomInfo? {
        if (values.size < ROOM_INFO_FIELD_COUNT) return null
        return RoomInfo(values[0], values[1], values[2], values[3], values[4], values[5])
    }

    private fun parseMembers(values: Array<String>): List<RoomMember> = values.toList()
        .chunked(MEMBER_FIELD_COUNT)
        .filter { it.size == MEMBER_FIELD_COUNT }
        .map { RoomMember(it[0], it[1], it[2], it[3]) }

    private fun parseEvents(values: Array<String>): List<RoomEvent> = values.toList()
        .chunked(EVENT_FIELD_COUNT)
        .filter { it.size == EVENT_FIELD_COUNT }
        .mapNotNull { event ->
            if (event[0] == EVENT_CHAT) {
                RoomEvent.Chat(event[1], event[2], event[3])
            } else {
                val type = when (event[3].toIntOrNull()) {
                    1 -> RoomEvent.Status.Type.JOINED
                    2 -> RoomEvent.Status.Type.LEFT
                    3 -> RoomEvent.Status.Type.KICKED
                    4 -> RoomEvent.Status.Type.BANNED
                    5 -> RoomEvent.Status.Type.UNBANNED
                    else -> null
                }
                type?.let { RoomEvent.Status(event[1], event[2], it) }
            }
        }

    private const val ROOM_INFO_FIELD_COUNT = 6
    private const val MEMBER_FIELD_COUNT = 4
    private const val EVENT_FIELD_COUNT = 4
    private const val EVENT_CHAT = "0"
    private const val ERROR_POLL_ATTEMPTS = 5
    private const val ERROR_POLL_DELAY_MS = 20L
}
