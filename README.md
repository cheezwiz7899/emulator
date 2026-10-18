# Citron Neo

<img src="./dist/citron.svg" width="128" alt="Citron Neo logo">

Citron Neo is an open-source emulator that continues the Citron codebase. It
runs on Windows, Linux, macOS and Android (arm64). Development and maintenance
happen on [Discord](https://discord.gg/7bdSQgQZfu).

## Building

Library dependencies are fetched at configure time by CPM, so you don't need to
initialise submodules first.

**Linux.** Run `./build-citron-linux.sh setup` once, then
`./build-citron-linux.sh use --pgo none --lto full` for a build without the
profiling round. The binary lands in `build/use-nopgo/bin/citron`, with an
AppImage beside it. [docs/BUILDING-CITRON-LINUX.md](docs/BUILDING-CITRON-LINUX.md)
covers the PGO stages, which give the best performance.

**Windows.** Build with clang-cl natively, or cross-compile from Linux with
llvm-mingw. [docs/BUILDING-CITRON-WINDOWS.md](docs/BUILDING-CITRON-WINDOWS.md)
has both.

**Android.** From `src/android`, run `./gradlew assembleMainlineRelease`.
Without a release keystore in the environment, the APK is signed with the
bundled debug key.

## Contributing

Issues and pull requests are welcome. Ask on Discord before starting something
large. When reporting a bug, say which platform, GPU and driver you're on, and
which commit or build you're running.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE). Third-party components and their
licences are in [LICENSES/](LICENSES/).
