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
import org.citron.citron_emu.databinding.DialogDirectConnectBinding
import org.citron.citron_emu.features.multiplayer.data.MultiplayerPreferences
import org.citron.citron_emu.features.multiplayer.model.DirectConnectParams
import org.citron.citron_emu.features.multiplayer.model.MultiplayerSnapshot
import org.citron.citron_emu.features.multiplayer.model.MultiplayerValidation
import org.citron.citron_emu.features.multiplayer.model.RoomConnectionState

class DirectConnectDialogFragment : DialogFragment() {
    private lateinit var binding: DialogDirectConnectBinding
    private val multiplayerViewModel: MultiplayerViewModel by activityViewModels()
    private var stateJob: Job? = null
    private var connectJob: Job? = null
    private var cleanupJob: Job? = null
    private var validationShown = false
    private var awaitingConnection = false
    private var observedConnectionProgress = false
    private var pendingConnectedResult = false

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        binding = DialogDirectConnectBinding.inflate(layoutInflater)
        val saved = MultiplayerPreferences.loadDirectConnect()
        binding.directConnectHost.setText(saved.host)
        binding.directConnectPort.setText(saved.port.toString())
        binding.directConnectNickname.setText(saved.nickname)
        binding.directConnectHost.doAfterTextChanged {
            if (validationShown) binding.directConnectHostLayout.error = hostError(value = it?.toString()?.trim().orEmpty())
        }
        binding.directConnectPort.doAfterTextChanged {
            if (validationShown) binding.directConnectPortLayout.error = portError(it?.toString()?.toIntOrNull())
        }
        binding.directConnectNickname.doAfterTextChanged {
            if (validationShown) binding.directConnectNicknameLayout.error = identifierError(it?.toString()?.trim().orEmpty())
        }

        return MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.direct_connect)
            .setView(binding.root)
            .setNegativeButton(R.string.close, null)
            .setNeutralButton(R.string.direct_connect_disconnect, null)
            .setPositiveButton(R.string.direct_connect_connect, null)
            .create()
            .also { dialog ->
                dialog.setOnShowListener {
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                        connect(dialog)
                    }
                    dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                        disconnect(dialog, dismissWhenComplete = false)
                    }
                    dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                        val snapshot = multiplayerViewModel.snapshot.value
                        if (
                            awaitingConnection ||
                            snapshot.connectionState == RoomConnectionState.JOINING ||
                            cleanupJob?.isActive == true
                        ) {
                            disconnect(dialog, dismissWhenComplete = true)
                        } else {
                            dismiss()
                        }
                    }
                    observeState(dialog)
                    renderState(dialog, multiplayerViewModel.snapshot.value)
                }
            }
    }

    override fun onDismiss(dialog: DialogInterface) {
        if (awaitingConnection) {
            awaitingConnection = false
            pendingConnectedResult = false
            connectJob?.cancel()
            cleanupJob = multiplayerViewModel.leaveOrCloseInBackground(
                multiplayerViewModel.snapshot.value.isHosting
            )
        }
        saveForm()
        super.onDismiss(dialog)
    }

    override fun onResume() {
        super.onResume()
        completePendingConnectedTransition()
    }

    override fun onDestroy() {
        stateJob?.cancel()
        connectJob?.cancel()
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

    private fun connect(dialog: AlertDialog) {
        val params = currentParams()
        if (!validateForm(params.host, params.nickname, params.port)) {
            showStatus(R.string.direct_connect_invalid_input)
            return
        }

        awaitingConnection = true
        observedConnectionProgress = false
        pendingConnectedResult = false
        isCancelable = false
        showStatus(R.string.direct_connect_connecting)
        renderState(dialog, multiplayerViewModel.snapshot.value)
        connectJob?.cancel()
        connectJob = lifecycleScope.launch {
            val result = multiplayerViewModel.connect(params)
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
                showStatus(result.error?.messageId ?: R.string.multiplayer_error_unknown)
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
                pendingConnectedResult = true
                completePendingConnectedTransition()
            }
            snapshot.connectionState == RoomConnectionState.JOINING -> {
                observedConnectionProgress = true
            }
            observedConnectionProgress && snapshot.connectionState == RoomConnectionState.IDLE -> {
                awaitingConnection = false
                pendingConnectedResult = false
                isCancelable = true
                showStatus(snapshot.lastError?.messageId ?: R.string.multiplayer_error_unknown)
                renderState(dialog, snapshot)
            }
        }
    }

    private fun completePendingConnectedTransition() {
        if (
            !pendingConnectedResult ||
            !lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED) ||
            parentFragmentManager.isStateSaved
        ) {
            return
        }
        pendingConnectedResult = false
        awaitingConnection = false
        dismissAllowingStateLoss()
        if (parentFragmentManager.findFragmentByTag(RoomDialogFragment.TAG) == null) {
            RoomDialogFragment().show(parentFragmentManager, RoomDialogFragment.TAG)
        }
    }

    private fun disconnect(dialog: AlertDialog, dismissWhenComplete: Boolean) {
        val snapshot = multiplayerViewModel.snapshot.value
        awaitingConnection = false
        pendingConnectedResult = false
        isCancelable = false
        connectJob?.cancel()
        val activeCleanup = cleanupJob?.takeIf { it.isActive }
        val cleanup = activeCleanup ?: multiplayerViewModel.leaveOrCloseInBackground(
            snapshot.isHosting
        ).also { cleanupJob = it }
        renderState(dialog, snapshot)
        lifecycleScope.launch {
            cleanup.join()
            if (!isAdded) return@launch
            cleanupJob = null
            isCancelable = true
            if (dismissWhenComplete) {
                dismissAllowingStateLoss()
                return@launch
            }
            showStatus(R.string.direct_connect_disconnected)
            renderState(dialog, multiplayerViewModel.snapshot.value)
        }
    }

    private fun renderState(dialog: AlertDialog, snapshot: MultiplayerSnapshot) {
        val connecting = snapshot.connectionState == RoomConnectionState.JOINING
        val connected = snapshot.isConnected
        val cleaningUp = cleanupJob?.isActive == true
        val busy = connecting || connected || awaitingConnection || cleaningUp
        binding.directConnectHost.isEnabled = !busy
        binding.directConnectPort.isEnabled = !busy
        binding.directConnectNickname.isEnabled = !busy
        binding.directConnectPassword.isEnabled = !busy
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = snapshot.canStartConnection &&
            !awaitingConnection && !cleaningUp
        dialog.getButton(AlertDialog.BUTTON_NEUTRAL).isEnabled = !cleaningUp &&
            (connecting || connected || awaitingConnection)
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).isEnabled = true

        when {
            connecting -> showStatus(R.string.direct_connect_connecting)
            connected -> showStatus(R.string.direct_connect_connected)
        }
    }

    private fun currentParams() = DirectConnectParams(
        host = binding.directConnectHost.text?.toString()?.trim().orEmpty(),
        port = binding.directConnectPort.text?.toString()?.toIntOrNull() ?: 0,
        nickname = binding.directConnectNickname.text?.toString()?.trim().orEmpty(),
        password = binding.directConnectPassword.text?.toString().orEmpty()
    )

    private fun saveForm() {
        MultiplayerPreferences.saveDirectConnectForm(
            MultiplayerPreferences.DirectConnectForm(
                binding.directConnectHost.text?.toString()?.trim().orEmpty(),
                binding.directConnectPort.text?.toString()?.toIntOrNull() ?: 0,
                binding.directConnectNickname.text?.toString()?.trim().orEmpty()
            )
        )
    }

    private fun validateForm(host: String, nickname: String, port: Int?): Boolean {
        validationShown = true
        binding.directConnectHostLayout.error = hostError(host)
        binding.directConnectPortLayout.error = portError(port)
        binding.directConnectNicknameLayout.error = identifierError(nickname)
        return binding.directConnectHostLayout.error == null &&
            binding.directConnectPortLayout.error == null &&
            binding.directConnectNicknameLayout.error == null
    }

    private fun hostError(value: String): String? =
        if (MultiplayerValidation.isValidHost(value)) null else getString(R.string.multiplayer_invalid_host)

    private fun portError(value: Int?): String? =
        if (MultiplayerValidation.isValidPort(value)) null else getString(R.string.multiplayer_invalid_port)

    private fun identifierError(value: String): String? =
        if (MultiplayerValidation.isValidIdentifier(value)) null else getString(R.string.multiplayer_invalid_identifier)

    private fun showStatus(stringId: Int) {
        binding.directConnectStatus.setText(stringId)
        binding.directConnectStatus.isVisible = true
    }

    companion object {
        const val TAG = "DirectConnectDialog"
    }
}
