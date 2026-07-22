# Build Modernization

This repo now has an additive CMake path for native C++ tests. The existing MSBuild projects and scripts remain the authoritative application build until DML, CUDA, and worker parity are proven.

## Current Authoritative Builds

- `build-ninja.bat dml`
- `build-ninja.bat cuda`
- `build-ninja.bat worker`
- `build-ninja.bat all`

These still route through the existing Visual Studio projects and PowerShell dependency resolution.

## New Parallel CMake Path

The CMake surface builds native regression tests for targeting math and the first production pipeline seams:

- `include/aim_kalman.h`
- `include/aim_imm.h`
- `include/ego_motion_compensator.h`
- `tests/cpp/targeting_convergence_tests.cpp`
- `tests/cpp/unreal_synthetic_targeting_tests.cpp`
- `capture/capture_geometry.h`
- `detector/postProcess.cpp`
- `mouse/BoxTarget.cpp`

Configure, build, and test:

```powershell
cmake --preset vs2026-cuda-tests
cmake --build --preset vs2026-cuda-tests-debug
ctest --preset vs2026-cuda-tests-debug
```

This is still intentionally additive. It gives the project compiled C++ regression coverage for the runtime math, capture coordinate mapping, postprocess decoding, and tracker state transitions without making CMake responsible for the DML/CUDA application binaries yet.

## Dependency Strategy

- Keep TensorRT as an explicit SDK/imported target. NVIDIA still distributes Windows TensorRT as a zip/SDK install, so a fully package-managed TensorRT path is not assumed.
- Keep the current OpenCV 5 CUDA install path until package support matches the project requirement exactly. The native CMake tests discover that install so geometry, postprocess, and tracker tests compile against the same OpenCV headers/libraries as the app.
- Use `vs2026-cuda-tests` as the primary native test preset so dependency discovery validates CUDA, TensorRT, OpenCV CUDA, CUDA runtime DLLs, and cuDNN runtime DLLs before the tests run.
- Use vcpkg/CMake packages where they reduce friction without changing runtime behavior.
- Keep `packages.config` for DirectML/ONNX Runtime until the DML app target is migrated and validated under CMake.

## Test Strategy

- Keep Python contract tests. They protect architectural boundaries such as removed FOV/profile logic, direct tracker-to-mouse handoff, script invariants, and stale subsystem cleanup.
- Add native C++ tests for runtime math and stateful logic. Current native targets cover Kalman, IMM, ego-motion compensation, convergence tuning, Unreal-style synthetic targeting scenarios, capture geometry, YOLO/DML postprocess decoding, class-aware NMS, and tracker lock/lost/confirmed transitions.
- Keep convergence defaults grounded in measured behavior: the native convergence suite protects the seeded acquisition ramp, steadier target stream response, and reversal recovery checks that drove the current defaults.
- Add `clang-tidy` through CMake behind `OBS2_ENABLE_CLANG_TIDY=ON`. It is opt-in because not every Windows developer machine has LLVM installed.

## Migration Phases

1. Maintain current MSBuild scripts while adding CMake-only native tests.
2. Move shared source lists into CMake library targets without producing app binaries.
3. Add CMake DML app target and compare output/runtime behavior against `build-ninja.bat dml`.
4. Add CMake CUDA and worker targets with explicit TensorRT/OpenCV/CUDA imported dependencies.
5. Once parity is proven, make CMake the default build and retire duplicated `.vcxproj` maintenance.
