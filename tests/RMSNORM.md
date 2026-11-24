RMSNorm validation
==================

Build and run the active project tests (including CPU and CUDA RMSNorm):

```sh
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Build and run only the RMSNorm test executable (CMake also builds its
`kuiper` library dependency):

```sh
cmake -S . -B build
cmake --build build --target kuiper_rmsnorm_tests -j4
./build/kuiper_rmsnorm_tests
```

CPU computes each row using double accumulation with epsilon 1e-6.
The flattened entry point treats the entire tensor as one row; the dim entry
point normalizes the last dimension with one shared weight vector.
CUDA comparisons cover widths 1, 3, 31, 32, 33, 255, 256, 257, 480, 1024,
4099 and 72480, single/multiple rows, in-place output and default/nonblocking
streams. Each comparison synchronizes its stream before copying to the host.
Tolerance: 2e-5 * (1 + abs(CPU result)).

Sanitizer build:

```sh
cmake -S . -B build-sanitize -DKUIPER_ENABLE_SANITIZERS=ON
cmake --build build-sanitize -j4
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
```

Leak detection is disabled above for environments where LeakSanitizer cannot
inspect threads under ptrace. Host sanitizers do not check CUDA device accesses.
CUDA tests skip when a compatible CUDA device/driver is unavailable. On such
an environment a successful build does not establish GPU numerical correctness.
The legacy test/ tree references missing model/operator implementations and old
Tensor APIs; the active root CMake test target uses tests/.
