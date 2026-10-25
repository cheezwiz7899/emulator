Standalone checks of the ScarletNX CPU backend's moving parts against a real
`libscarletnx-recompiler` (not part of the Citron build). `shim/` fakes the few Citron headers they
need and must come first on the include path; the real sources under `src/core/arm/scarlet/` are
what is compiled.

    PREFIX=/opt/scarletnx tools/scarlet-harness/build.sh      # from the repo root; Linux x86-64

- `test.cpp`          GuestCoroutine: SVC/poll yield + resume, cross-thread resume, fresh entry at a non-leader pc
- `test_manager.cpp`  CodeManager: secondary modules (NRO-style code outside the loaded segments), invalidation, concurrent compile
- `test_fault.cpp`    the real scarlet_fault_handler.cpp: GPU-tracked page retry, real guest faults parked on the coroutine
