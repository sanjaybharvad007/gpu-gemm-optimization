#!/usr/bin/env bash
# Type-checks every CUDA source with the host C++ compiler (no nvcc / GPU needed).  Used by CI.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX="${CXX:-g++}"
status=0
for f in src/kernels/*.cu src/*.cu tests/cuda/*.cu; do
  if $CXX -fsyntax-only -std=c++17 -Wall -Wextra -Werror -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-but-set-variable \
       -x c++ -Iinclude -Isrc -Isrc/host_stub -DGEMM_HOST_STUB "$f"; then
    echo "ok   $f"
  else
    echo "FAIL $f"; status=1
  fi
done
exit $status
