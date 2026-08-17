// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.ui

import androidx.annotation.StringRes
import org.citron.citron_emu.R
import org.citron.citron_emu.features.multiplayer.model.MultiplayerError

@get:StringRes
val MultiplayerError.messageId: Int
    get() = when (this) {
        MultiplayerError.LOST_CONNECTION -> R.string.multiplayer_error_lost_connection
        MultiplayerError.KICKED -> R.string.multiplayer_error_kicked
        MultiplayerError.NAME_COLLISION -> R.string.multiplayer_error_name_collision
        MultiplayerError.IP_COLLISION -> R.string.multiplayer_error_ip_collision
        MultiplayerError.WRONG_VERSION -> R.string.multiplayer_error_wrong_version
        MultiplayerError.WRONG_PASSWORD -> R.string.multiplayer_error_wrong_password
        MultiplayerError.COULD_NOT_CONNECT -> R.string.multiplayer_error_could_not_connect
        MultiplayerError.ROOM_FULL -> R.string.multiplayer_error_room_full
        MultiplayerError.BANNED -> R.string.multiplayer_error_banned
        MultiplayerError.PERMISSION_DENIED -> R.string.multiplayer_error_permission_denied
        MultiplayerError.NO_SUCH_USER -> R.string.multiplayer_error_no_such_user
        MultiplayerError.NETWORK_NOT_INITIALIZED ->
            R.string.multiplayer_error_network_not_initialized
        MultiplayerError.INVALID_ARGUMENTS -> R.string.multiplayer_error_invalid_arguments
        MultiplayerError.NO_NETWORK_INTERFACE -> R.string.multiplayer_error_no_network_interface
        MultiplayerError.ROOM_UNAVAILABLE -> R.string.multiplayer_error_room_unavailable
        MultiplayerError.ROOM_ALREADY_OPEN -> R.string.multiplayer_error_room_already_open
        MultiplayerError.MEMBER_BUSY -> R.string.multiplayer_error_member_busy
        MultiplayerError.COULD_NOT_CREATE_ROOM -> R.string.multiplayer_error_could_not_create_room
        MultiplayerError.LOCAL_JOIN_FAILED -> R.string.multiplayer_error_local_join_failed
        MultiplayerError.UNKNOWN -> R.string.multiplayer_error_unknown
    }
