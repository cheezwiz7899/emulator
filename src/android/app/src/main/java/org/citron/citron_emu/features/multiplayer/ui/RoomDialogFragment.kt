// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.ui

import android.app.Dialog
import android.os.Bundle
import android.view.WindowManager
import androidx.appcompat.app.AlertDialog
import androidx.core.widget.doAfterTextChanged
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.citron.citron_emu.R
import org.citron.citron_emu.databinding.DialogMultiplayerRoomBinding
import org.citron.citron_emu.features.multiplayer.model.MultiplayerSnapshot
import org.citron.citron_emu.features.multiplayer.model.MultiplayerValidation
import org.citron.citron_emu.features.multiplayer.model.RoomContent
import org.citron.citron_emu.features.multiplayer.model.RoomEvent

class RoomDialogFragment : DialogFragment() {
    private lateinit var binding: DialogMultiplayerRoomBinding
    private val multiplayerViewModel: MultiplayerViewModel by activityViewModels()
    private var stateJob: Job? = null
    private var contentJob: Job? = null
    private val chatLines = ArrayDeque<String>()
    private var wasConnected = false
    private var renderedEventCount = 0

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        binding = DialogMultiplayerRoomBinding.inflate(layoutInflater)
        val snapshot = multiplayerViewModel.snapshot.value
        wasConnected = snapshot.isConnected
        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.room_title)
            .setView(binding.root)
            .setNegativeButton(
                if (snapshot.isHosting) R.string.room_close_hosted else R.string.room_leave,
                null
            )
            .setPositiveButton(R.string.close, null)
            .create()
            .also { dialog ->
                dialog.setOnShowListener {
                    enableImeResize(dialog)
                    val disconnectButton = dialog.getButton(AlertDialog.BUTTON_NEGATIVE)
                    disconnectButton.setOnClickListener {
                        disconnectButton.isEnabled = false
                        disconnect(multiplayerViewModel.snapshot.value.isHosting)
                    }
                    binding.roomSend.setOnClickListener { sendMessage() }
                    binding.roomMessage.doAfterTextChanged {
                        val message = it?.toString()?.trim().orEmpty()
                        binding.roomMessageLayout.error = if (
                            message.isEmpty() || MultiplayerValidation.isValidChatMessage(message)
                        ) {
                            null
                        } else {
                            getString(R.string.room_message_too_long)
                        }
                    }
                    binding.roomMessage.setOnEditorActionListener { _, _, _ ->
                        sendMessage()
                        true
                    }
                    observeState(dialog)
                    observeRoomContent()
                }
            }
    }

    @Suppress("DEPRECATION")
    private fun enableImeResize(dialog: AlertDialog) {
        dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
    }

    override fun onDestroy() {
        stateJob?.cancel()
        contentJob?.cancel()
        super.onDestroy()
    }

    private fun observeState(dialog: AlertDialog) {
        stateJob?.cancel()
        stateJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                multiplayerViewModel.snapshot.collect { renderState(dialog, it) }
            }
        }
    }

    private fun observeRoomContent() {
        contentJob?.cancel()
        contentJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                multiplayerViewModel.roomContent.collect(::renderRoomContent)
            }
        }
    }

    private fun renderState(dialog: AlertDialog, snapshot: MultiplayerSnapshot) {
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setText(
            if (snapshot.isHosting) R.string.room_close_hosted else R.string.room_leave
        )
        if (snapshot.isConnected) {
            wasConnected = true
            binding.roomMessage.isEnabled = true
            binding.roomSend.isEnabled = true
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).isEnabled = true
            return
        }

        binding.roomStatus.setText(
            snapshot.lastError?.messageId ?: if (wasConnected) {
                R.string.direct_connect_disconnected
            } else {
                R.string.direct_connect_connecting
            }
        )
        binding.roomMessage.isEnabled = false
        binding.roomSend.isEnabled = false
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).isEnabled = snapshot.isHosting
    }

    private fun renderRoomContent(content: RoomContent) {
        content.info?.let { info ->
            binding.roomSummary.text = getString(
                R.string.room_summary,
                info.name,
                info.memberCount,
                info.memberLimit
            )
            binding.roomDescription.text = info.description.ifBlank {
                getString(R.string.room_no_description)
            }
            binding.roomStatus.text = if (info.gameName.isBlank()) {
                getString(R.string.room_connected_port, info.port)
            } else {
                getString(R.string.room_connected_game, info.gameName, info.port)
            }
        }

        binding.roomMembers.text = content.members.joinToString("\n") { member ->
            val displayName = displayName(member.nickname, member.username)
            when {
                member.gameName.isBlank() -> displayName
                member.gameId.isBlank() -> "$displayName — ${member.gameName}"
                else -> "$displayName — ${member.gameName} (${member.gameId})"
            }
        }
        if (content.events.size < renderedEventCount) renderedEventCount = 0
        content.events.drop(renderedEventCount).forEach(::appendEvent)
        renderedEventCount = content.events.size
    }

    private fun appendEvent(event: RoomEvent) {
        val displayName = displayName(event.nickname, event.username)
        val line = when (event) {
            is RoomEvent.Chat -> getString(R.string.room_chat_line, displayName, event.message)
            is RoomEvent.Status -> when (event.type) {
                RoomEvent.Status.Type.JOINED -> getString(R.string.room_member_joined, displayName)
                RoomEvent.Status.Type.LEFT -> getString(R.string.room_member_left, displayName)
                RoomEvent.Status.Type.KICKED -> getString(R.string.room_member_kicked, displayName)
                RoomEvent.Status.Type.BANNED -> getString(R.string.room_member_banned, displayName)
                RoomEvent.Status.Type.UNBANNED -> getString(R.string.room_member_unbanned, displayName)
            }
        }
        appendChatLine(line)
    }

    private fun displayName(nickname: String, username: String): String =
        if (username.isBlank() || username == nickname) nickname else "$nickname ($username)"

    private fun sendMessage() {
        val message = binding.roomMessage.text?.toString()?.trim().orEmpty()
        if (message.isEmpty()) return
        if (!MultiplayerValidation.isValidChatMessage(message)) {
            binding.roomMessageLayout.error = getString(R.string.room_message_too_long)
            return
        }
        binding.roomSend.isEnabled = false
        lifecycleScope.launch {
            val sent = multiplayerViewModel.sendChatMessage(message)
            binding.roomSend.isEnabled = multiplayerViewModel.snapshot.value.isConnected
            if (sent) {
                appendChatLine(getString(R.string.room_chat_line_self, message))
                binding.roomMessage.text?.clear()
            } else {
                binding.roomMessageLayout.error = getString(R.string.room_message_send_failed)
            }
        }
    }

    private fun appendChatLine(line: String) {
        chatLines.addLast(line)
        while (chatLines.size > MAX_VISIBLE_EVENTS) chatLines.removeFirst()
        binding.roomChatHistory.text = chatLines.joinToString("\n")
        binding.roomChatScroll.post {
            binding.roomChatScroll.fullScroll(android.view.View.FOCUS_DOWN)
        }
    }

    private fun disconnect(hosting: Boolean) {
        stateJob?.cancel()
        contentJob?.cancel()
        lifecycleScope.launch {
            multiplayerViewModel.leaveOrClose(hosting)
            dismissAllowingStateLoss()
        }
    }

    companion object {
        const val TAG = "RoomDialog"
        private const val MAX_VISIBLE_EVENTS = 100
    }
}
