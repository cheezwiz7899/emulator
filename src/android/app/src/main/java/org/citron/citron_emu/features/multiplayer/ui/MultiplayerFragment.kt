// SPDX-License-Identifier: GPL-2.0-or-later

package org.citron.citron_emu.features.multiplayer.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch
import org.citron.citron_emu.R
import org.citron.citron_emu.databinding.FragmentMultiplayerBinding
import org.citron.citron_emu.features.multiplayer.model.MultiplayerSnapshot
import org.citron.citron_emu.features.multiplayer.model.RoomConnectionState
import org.citron.citron_emu.model.HomeViewModel
import org.citron.citron_emu.utils.ViewUtils.updateMargins

class MultiplayerFragment : Fragment() {
    private var _binding: FragmentMultiplayerBinding? = null
    private val binding get() = _binding!!
    private val homeViewModel: HomeViewModel by activityViewModels()
    private val multiplayerViewModel: MultiplayerViewModel by activityViewModels()
    private var updatingAirplaneSwitch = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMultiplayerBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        homeViewModel.setNavigationVisibility(visible = true, animated = true)
        homeViewModel.setStatusBarShadeVisibility(visible = true)

        binding.multiplayerAirplaneMode.setOnCheckedChangeListener { _, enabled ->
            if (!updatingAirplaneSwitch) {
                multiplayerViewModel.setAirplaneModeEnabled(enabled)
            }
        }
        binding.multiplayerDirectConnect.setOnClickListener {
            if (parentFragmentManager.findFragmentByTag(DirectConnectDialogFragment.TAG) == null) {
                DirectConnectDialogFragment().show(
                    parentFragmentManager,
                    DirectConnectDialogFragment.TAG
                )
            }
        }
        binding.multiplayerCreateRoom.setOnClickListener {
            if (parentFragmentManager.findFragmentByTag(HostRoomDialogFragment.TAG) == null) {
                HostRoomDialogFragment().show(parentFragmentManager, HostRoomDialogFragment.TAG)
            }
        }
        binding.multiplayerOpenRoom.setOnClickListener {
            if (parentFragmentManager.findFragmentByTag(RoomDialogFragment.TAG) == null) {
                RoomDialogFragment().show(parentFragmentManager, RoomDialogFragment.TAG)
            }
        }
        binding.multiplayerLeaveRoom.setOnClickListener { leaveOrCloseRoom() }

        setInsets()
        observeMultiplayerState()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun observeMultiplayerState() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch {
                    multiplayerViewModel.snapshot.collect(::renderSnapshot)
                }
                launch {
                    multiplayerViewModel.airplaneMode.collect { enabled ->
                        if (binding.multiplayerAirplaneMode.isChecked != enabled) {
                            updatingAirplaneSwitch = true
                            binding.multiplayerAirplaneMode.isChecked = enabled
                            updatingAirplaneSwitch = false
                        }
                    }
                }
            }
        }
    }

    private fun renderSnapshot(snapshot: MultiplayerSnapshot) {
        binding.multiplayerStatusText.setText(
            when {
                snapshot.isHosting && snapshot.isConnected -> R.string.multiplayer_status_hosting
                snapshot.isHosting -> R.string.multiplayer_status_host_disconnected
                snapshot.connectionState == RoomConnectionState.JOINING ->
                    R.string.multiplayer_status_connecting
                snapshot.isConnected -> R.string.multiplayer_status_connected
                else -> R.string.multiplayer_status_disconnected
            }
        )

        binding.multiplayerDirectConnect.isEnabled = snapshot.canStartConnection
        binding.multiplayerCreateRoom.isEnabled = snapshot.canStartConnection
        binding.multiplayerOpenRoom.isEnabled = snapshot.isConnected || snapshot.isHosting
        binding.multiplayerLeaveRoom.isEnabled = snapshot.isConnected || snapshot.isHosting
        binding.multiplayerAirplaneMode.isEnabled = snapshot.canStartConnection
        binding.multiplayerLeaveRoom.setText(
            if (snapshot.isHosting) R.string.room_close_hosted else R.string.room_leave
        )
    }

    private fun leaveOrCloseRoom() {
        val hosting = multiplayerViewModel.snapshot.value.isHosting
        viewLifecycleOwner.lifecycleScope.launch {
            multiplayerViewModel.leaveOrClose(hosting)
            Toast.makeText(
                requireContext(),
                if (hosting) {
                    R.string.multiplayer_close_complete
                } else {
                    R.string.multiplayer_leave_complete
                },
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    private fun setInsets() = ViewCompat.setOnApplyWindowInsetsListener(binding.root) {
            view: View,
            windowInsets: WindowInsetsCompat ->
        val barInsets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
        val cutoutInsets = windowInsets.getInsets(WindowInsetsCompat.Type.displayCutout())
        val spacingNavigation = resources.getDimensionPixelSize(R.dimen.spacing_navigation)
        val spacingNavigationRail =
            resources.getDimensionPixelSize(R.dimen.spacing_navigation_rail)
        val contentPadding = resources.getDimensionPixelSize(R.dimen.spacing_fab)

        binding.multiplayerScroll.updateMargins(
            left = barInsets.left + cutoutInsets.left,
            right = barInsets.right + cutoutInsets.right
        )
        binding.multiplayerScroll.updatePadding(top = barInsets.top, bottom = barInsets.bottom)
        binding.multiplayerContent.updatePadding(bottom = spacingNavigation + contentPadding)
        if (view.layoutDirection == View.LAYOUT_DIRECTION_LTR) {
            binding.multiplayerContent.updatePadding(left = spacingNavigationRail + contentPadding)
        } else {
            binding.multiplayerContent.updatePadding(right = spacingNavigationRail + contentPadding)
        }
        windowInsets
    }
}
