#!/bin/sh
# Builds and runs the standalone checks against an installed libscarletnx-recompiler.
# usage: PREFIX=/opt/scarletnx tools/scarlet-harness/build.sh   (run from the repo root)
set -e
PREFIX=${PREFIX:-/opt/scarletnx}
H=tools/scarlet-harness
INC="-I$H/shim -Isrc -I$PREFIX/include -I$PREFIX/include/scarletnx"
LIBS="$PREFIX/lib/libscarletnx-recompiler.a $(llvm-config --ldflags --libs --system-libs) -lpthread -ldl -lboost_context"
MGR=src/core/arm/scarlet/scarlet_code_manager.cpp
FLT=src/core/arm/scarlet/scarlet_fault_handler.cpp
OUT=${OUT:-/tmp/scarlet-harness}; mkdir -p "$OUT"
clang++ -std=c++20 -O1 -DSCARLET_HARNESS $INC $H/test.cpp $LIBS -o "$OUT/test_coroutine"
clang++ -std=c++20 -O1 -DSCARLET_HARNESS $INC $H/test_manager.cpp $MGR $LIBS -o "$OUT/test_manager"
clang++ -std=c++20 -O1 -DSCARLET_HARNESS $INC $H/test_fault.cpp $MGR $FLT $LIBS -o "$OUT/test_fault"
for t in test_coroutine test_manager test_fault; do echo "== $t"; "$OUT/$t" 2>&1 | grep -E "^(test|call|after|4 threads|tracked|untracked|RESULT|FAIL)"; done
