// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.ui

import android.app.Dialog
import android.content.DialogInterface
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.view.isVisible
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
import org.citron.citron_emu.databinding.DialogHostRoomBinding
import org.citron.citron_emu.features.multiplayer.data.MultiplayerPreferences
import org.citron.citron_emu.features.multiplayer.model.HostRoomParams
import org.citron.citron_emu.features.multiplayer.model.MultiplayerSnapshot
import org.citron.citron_emu.features.multiplayer.model.MultiplayerValidation
import org.citron.citron_emu.features.multiplayer.model.RoomConnectionState

class HostRoomDialogFragment : DialogFragment() {
    private lateinit var binding: DialogHostRoomBinding
    private val multiplayerViewModel: MultiplayerViewModel by activityViewModels()
    private var stateJob: Job? = null
    private var hostJob: Job? = null
    private var validationShown = false
    private var awaitingConnection = false
    private var observedConnectionProgress = false
    private var cleanupInProgress = false

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        binding = DialogHostRoomBinding.inflate(layoutInflater)
        val saved = MultiplayerPreferences.loadHostRoom()
        binding.hostRoomName.setText(saved.roomName)
        binding.hostRoomNickname.setText(saved.nickname)
        binding.hostRoomDescription.setText(saved.description)
        binding.hostRoomPort.setText(saved.port.toString())
        binding.hostRoomMaxPlayers.setText(saved.maxPlayers.toString())
        binding.hostRoomName.doAfterTextChanged {
            if (validationShown) {
                binding.hostRoomNameLayout.error = identifierError(it?.toString()?.trim().orEmpty())
            }
        }
        binding.hostRoomNickname.doAfterTextChanged {
            if (validationShown) {
                binding.hostRoomNicknameLayout.error = identifierError(it?.toString()?.trim().orEmpty())
            }
        }
        binding.hostRoomPort.doAfterTextChanged {
            if (validationShown) binding.hostRoomPortLayout.error = portError(it?.toString()?.toIntOrNull())
        }
        binding.hostRoomMaxPlayers.doAfterTextChanged {
            if (validationShown) {
                binding.hostRoomMaxPlayersLayout.error = playerCountError(it?.toString()?.toIntOrNull())
            }
        }

        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.host_room_unlisted)
            .setView(binding.root)
            .setNegativeButton(R.string.close, null)
            .setPositiveButton(R.string.host_room_create, null)
            .create()
            .also { dialog ->
                dialog.setOnShowListener {
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                        host(dialog)
                    }
                    observeState(dialog)
                    renderState(dialog, multiplayerViewModel.snapshot.value)
                }
            }
    }

    override fun onDismiss(dialog: DialogInterface) {
        hostJob?.cancel()
        awaitingConnection = false
        isCancelable = true
        saveForm()
        super.onDismiss(dialog)
    }

    override fun onDestroy() {
        stateJob?.cancel()
        hostJob?.cancel()
        super.onDestroy()
    }

    private fun observeState(dialog: AlertDialog) {
        stateJob?.cancel()
        stateJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                multiplayerViewModel.snapshot.collect { snapshot ->
                    renderState(dialog, snapshot)
                    handleConnectionResult(dialog, snapshot)
                }
            }
        }
    }

    private fun host(dialog: AlertDialog) {
        val params = currentParams()
        if (!validateForm(params.roomName, params.nickname, params.port, params.maxPlayers)) {
            showError(getString(R.string.host_room_invalid_input))
            return
        }

        awaitingConnection = true
        observedConnectionProgress = false
        cleanupInProgress = false
        isCancelable = false
        showStatus(getString(R.string.host_room_creating))
        renderState(dialog, multiplayerViewModel.snapshot.value)
        hostJob?.cancel()
        hostJob = lifecycleScope.launch {
            val result = multiplayerViewModel.host(params)
            if (result.airplaneModeDisabled) {
                Toast.makeText(
                    requireContext(),
                    R.string.multiplayer_airplane_mode_disabled,
                    Toast.LENGTH_LONG
                ).show()
            }
            if (!result.started) {
                awaitingConnection = false
                isCancelable = true
                showError(getString(result.error?.messageId ?: R.string.multiplayer_error_unknown))
                renderState(dialog, multiplayerViewModel.snapshot.value)
            } else {
                observedConnectionProgress = true
                handleConnectionResult(
                    dialog,
                    result.snapshot ?: multiplayerViewModel.snapshot.value
                )
            }
        }
    }

    private fun handleConnectionResult(dialog: AlertDialog, snapshot: MultiplayerSnapshot) {
        if (!awaitingConnection) return
        when {
            snapshot.isConnected -> {
                awaitingConnection = false
                dismissAllowingStateLoss()
                RoomDialogFragment().show(parentFragmentManager, RoomDialogFragment.TAG)
            }
            snapshot.connectionState == RoomConnectionState.JOINING -> {
                observedConnectionProgress = true
            }
            observedConnectionProgress && snapshot.connectionState == RoomConnectionState.IDLE -> {
                if (cleanupInProgress) return
                showError(
                    getString(snapshot.lastError?.messageId ?: R.string.multiplayer_error_unknown)
                )
                if (snapshot.isHosting) {
                    cleanupInProgress = true
                    lifecycleScope.launch {
                        multiplayerViewModel.leaveOrClose(hosting = true)
                        cleanupInProgress = false
                        awaitingConnection = false
                        isCancelable = true
                        renderState(dialog, multiplayerViewModel.snapshot.value)
                    }
                } else {
                    awaitingConnection = false
                    isCancelable = true
                    renderState(dialog, snapshot)
                }
            }
        }
    }

    private fun renderState(dialog: AlertDialog, snapshot: MultiplayerSnapshot) {
        val busy = snapshot.connectionState == RoomConnectionState.JOINING || snapshot.isConnected ||
            awaitingConnection
        binding.hostRoomName.isEnabled = !busy
        binding.hostRoomNickname.isEnabled = !busy
        binding.hostRoomDescription.isEnabled = !busy
        binding.hostRoomPort.isEnabled = !busy
        binding.hostRoomPassword.isEnabled = !busy
        binding.hostRoomMaxPlayers.isEnabled = !busy
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = snapshot.canStartConnection &&
            !awaitingConnection
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).isEnabled = true
    }

    private fun currentParams() = HostRoomParams(
        roomName = binding.hostRoomName.text?.toString()?.trim().orEmpty(),
        nickname = binding.hostRoomNickname.text?.toString()?.trim().orEmpty(),
        description = binding.hostRoomDescription.text?.toString()?.trim().orEmpty(),
        port = binding.hostRoomPort.text?.toString()?.toIntOrNull() ?: 0,
        password = binding.hostRoomPassword.text?.toString().orEmpty(),
        maxPlayers = binding.hostRoomMaxPlayers.text?.toString()?.toIntOrNull() ?: 0
    )

    private fun saveForm() {
        val params = currentParams()
        MultiplayerPreferences.saveHostRoomForm(
            MultiplayerPreferences.HostRoomForm(
                params.roomName,
                params.nickname,
                params.description,
                params.port,
                params.maxPlayers
            )
        )
    }

    private fun validateForm(
        roomName: String,
        nickname: String,
        port: Int?,
        maxPlayers: Int?
    ): Boolean {
        validationShown = true
        binding.hostRoomNameLayout.error = identifierError(roomName)
        binding.hostRoomNicknameLayout.error = identifierError(nickname)
        binding.hostRoomPortLayout.error = portError(port)
        binding.hostRoomMaxPlayersLayout.error = playerCountError(maxPlayers)
        return binding.hostRoomNameLayout.error == null &&
            binding.hostRoomNicknameLayout.error == null &&
            binding.hostRoomPortLayout.error == null &&
            binding.hostRoomMaxPlayersLayout.error == null
    }

    private fun identifierError(value: String): String? =
        if (MultiplayerValidation.isValidIdentifier(value)) null else getString(R.string.multiplayer_invalid_identifier)

    private fun portError(value: Int?): String? =
        if (MultiplayerValidation.isValidPort(value)) null else getString(R.string.multiplayer_invalid_port)

    private fun playerCountError(value: Int?): String? =
        if (MultiplayerValidation.isValidPlayerCount(value)) null else getString(R.string.multiplayer_invalid_player_count)

    private fun showStatus(message: String) {
        binding.hostRoomStatus.text = message
        binding.hostRoomStatus.isVisible = true
    }

    private fun showError(message: String) {
        showStatus(message)
        Toast.makeText(requireContext(), message, Toast.LENGTH_LONG).show()
    }

    companion object {
        const val TAG = "HostRoomDialog"
    }
}
