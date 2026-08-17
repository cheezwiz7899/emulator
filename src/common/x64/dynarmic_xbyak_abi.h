// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// The interface dynarmic's DYNARMIC_XBYAK_CUSTOM_CONTAINERS branch includes by
// name (see dynarmic/common/x64/xbyak.h in the fork). Keeping it as its own
// file, rather than having dynarmic's fork include xbyak_abi.h/xbyak_util.h
// directly, so that other dynarmic users do not have to know or care about
// the split headers that citron neo uses.
//
// Previously this lived at src/dynarmic/common/x64/xbyak.h, mirroring the
// dynarmic path, so the fork's include of "dynarmic/common/x64/xbyak.h"
// would resolve here instead of to its own copy. That stopped working once
// the fork shipped a real file at that exact path: dynarmic's own include
// directory is searched first, so its copy started winning regardless of
// what this one contained. This path should be safe from collisions.
#pragma once
#include "common/x64/xbyak_abi.h"
#include "common/x64/xbyak_util.h"
