// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include "core/core.h"

TEST_CASE("System delivers an exit registered before the request", "[core]") {
    Core::System system;
    int calls = 0;
    auto observed_result = Core::SystemResultStatus::Success;
    system.RegisterExitCallback([&] {
        ++calls;
        observed_result = system.GetExitResult();
    });

    system.Exit(Core::SystemResultStatus::ErrorVideoCore);

    REQUIRE(calls == 1);
    REQUIRE(observed_result == Core::SystemResultStatus::ErrorVideoCore);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::ErrorVideoCore);
}

TEST_CASE("System delivers an exit requested before callback registration", "[core]") {
    Core::System system;
    int calls = 0;
    auto observed_result = Core::SystemResultStatus::Success;
    system.Exit(Core::SystemResultStatus::ErrorVideoCore);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::ErrorVideoCore);

    system.RegisterExitCallback([&] {
        ++calls;
        observed_result = system.GetExitResult();
    });

    REQUIRE(calls == 1);
    REQUIRE(observed_result == Core::SystemResultStatus::ErrorVideoCore);
    system.Exit(Core::SystemResultStatus::ErrorUnknown);
    REQUIRE(calls == 1);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::ErrorVideoCore);
}

TEST_CASE("Registering and requesting exit concurrently delivers once", "[core]") {
    Core::System system;
    std::atomic<int> calls = 0;
    std::atomic<bool> start = false;

    std::thread registration([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        system.RegisterExitCallback([&] { ++calls; });
    });
    std::thread exit([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        system.Exit(Core::SystemResultStatus::ErrorVideoCore);
    });
    start.store(true);
    registration.join();
    exit.join();

    REQUIRE(calls.load() == 1);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::ErrorVideoCore);
}

TEST_CASE("Concurrent exit requests deliver only one callback", "[core]") {
    Core::System system;
    std::atomic<int> calls = 0;
    system.RegisterExitCallback([&] { ++calls; });

    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] { system.Exit(Core::SystemResultStatus::ErrorVideoCore); });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    REQUIRE(calls.load() == 1);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::ErrorVideoCore);
}

TEST_CASE("Normal successful exit retains the normal result", "[core]") {
    Core::System system;
    int calls = 0;
    system.RegisterExitCallback([&] { ++calls; });

    system.Exit();
    system.Exit();

    REQUIRE(calls == 1);
    REQUIRE(system.GetExitResult() == Core::SystemResultStatus::Success);
}
