// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/maxwell/location.h"

namespace {

TEST_CASE("Maxwell CFG locations round-trip their encoded representation", "[video_core]") {
    // The normal constructor accepts a byte address and advances it to the
    // first instruction slot. Serialized CFG locations have already undergone
    // that adjustment and must not be passed through it a second time.
    REQUIRE(Shader::Maxwell::Location{32}.Offset() == 40);
    REQUIRE(Shader::Maxwell::Location::FromRawOffset(32).Offset() == 32);
    REQUIRE(Shader::Maxwell::Location::FromRawOffset(36).IsVirtual());
    REQUIRE(Shader::Maxwell::Location::IsRawOffset(32));
    REQUIRE(Shader::Maxwell::Location::IsRawOffset(36));
    REQUIRE_FALSE(Shader::Maxwell::Location::IsRawOffset(0xccccccccU));
}

} // Anonymous namespace
