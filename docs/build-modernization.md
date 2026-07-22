# Build Modernization

This repo now has an additive CMake path for native C++ tests. The existing MSBuild projects and scripts remain the authoritative application build until DML, CUDA, and worker parity are proven.

## Current Authoritative Builds

- `build-ninja.bat dml`
- `build-ninja.bat cuda`
- `build-ninja.bat worker`
- `build-ninja.bat all`

These still route through the existing Visual Studio projects and PowerShell dependency resolution.

## New Parallel CMake Path

The initial CMake surface builds pure targeting logic only:

- `include/aim_kalman.h`
- `include/aim_imm.h`
- `include/ego_motion_compensator.h`

Configure, build, and test:

```powershell
cmake --preset vs2026-tests
cmake --build --preset vs2026-tests-debug
ctest --preset vs2026-tests-debug
```

This is intentionally small. It gives the project compiled C++ regression coverage without pulling Win32 capture, ImGui, OpenCV, TensorRT, DirectML, serial devices, or CUDA into the first CMake milestone.

## Dependency Strategy

- Keep TensorRT as an explicit SDK/imported target. NVIDIA still distributes Windows TensorRT as a zip/SDK install, so a fully package-managed TensorRT path is not assumed.
- Keep the current OpenCV 5 CUDA install path until package support matches the project requirement exactly.
- Use vcpkg/CMake packages where they reduce friction without changing runtime behavior.
- Keep `packages.config` for DirectML/ONNX Runtime until the DML app target is migrated and validated under CMake.

## Test Strategy

- Keep Python contract tests. They protect architectural boundaries such as removed FOV/profile logic, direct tracker-to-mouse handoff, script invariants, and stale subsystem cleanup.
- Add native C++ tests for runtime math and stateful logic. The first native test target covers Kalman, IMM, and ego-motion compensation.
- Add `clang-tidy` through CMake behind `OBS2_ENABLE_CLANG_TIDY=ON`. It is opt-in because not every Windows developer machine has LLVM installed.

## Migration Phases

1. Maintain current MSBuild scripts while adding CMake-only native tests.
2. Move shared source lists into CMake library targets without producing app binaries.
3. Add CMake DML app target and compare output/runtime behavior against `build-ninja.bat dml`.
4. Add CMake CUDA and worker targets with explicit TensorRT/OpenCV/CUDA imported dependencies.
5. Once parity is proven, make CMake the default build and retire duplicated `.vcxproj` maintenance.
