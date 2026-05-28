# CUDA Build Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate CUDA/TensorRT build surface without changing the current DML project behavior.

**Architecture:** Keep the root `0BS_box_2.vcxproj` DML-only. Add a root-level `cuda/` folder with its own solution, project, filters, README, and build script. CUDA-only sources are compiled only by the CUDA project and output goes to `x64/CUDA`.

**Tech Stack:** Visual Studio/MSBuild, CUDA Toolkit 13.2 build customizations, TensorRT supplied through `TensorRTDir`, existing C++17 source tree.

---

### Task 1: Contract Test

**Files:**
- Create: `training/tests/test_cuda_build_isolation_contract.py`

- [x] Write a unittest that requires:
  - root DML project has no `CUDA|x64`, `USE_CUDA`, `CudaCompile`, or TensorRT/CUDA sources
  - `cuda/0BS_cuda.vcxproj`, `.filters`, `.sln`, `build-cuda.ps1`, and `README.md` exist
  - CUDA project defines `CUDA|x64`, `USE_CUDA`, CUDA 13.2 imports, CUDA source compile items, and `x64/CUDA` output

- [x] Run the test and confirm it fails before project files exist.

### Task 2: CUDA Folder

**Files:**
- Create: `cuda/0BS_cuda.vcxproj`
- Create: `cuda/0BS_cuda.vcxproj.filters`
- Create: `cuda/0BS_cuda.sln`
- Create: `cuda/build-cuda.ps1`
- Create: `cuda/README.md`

- [x] Add a separate CUDA project referencing source files via `..\`.
- [x] Add `USE_CUDA` and `CudaCompile` only to the CUDA project.
- [x] Keep root DML project unchanged.

### Task 3: Verification

**Commands:**

```powershell
python -m unittest training.tests.test_cuda_build_isolation_contract -v
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe' 0BS_box_2.vcxproj /p:Configuration=DML /p:Platform=x64 /m /v:minimal
.\cuda\build-cuda.ps1
```

**Expected:** Contract test passes. DML build passes. CUDA build may stop with an explicit TensorRT setup message if TensorRT is not installed or `TensorRTDir` is unset.
