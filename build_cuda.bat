@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto usage
if /I "%~1"=="/?" goto usage

pushd "%~dp0" || exit /b 1

where powershell >nul 2>nul
if errorlevel 1 (
    echo [build-cuda] ERROR: powershell was not found in PATH.
    goto fail
)

powershell -NoProfile -ExecutionPolicy Bypass -File "tools\build_cuda.ps1" %*
if errorlevel 1 goto fail

popd
exit /b 0

:fail
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if "%BUILD_EXIT_CODE%"=="0" set "BUILD_EXIT_CODE=1"
echo [build-cuda] Failed with exit code %BUILD_EXIT_CODE%.
pause
popd
exit /b %BUILD_EXIT_CODE%

:usage
echo Usage: build_cuda.bat [tools\build_cuda.ps1 arguments]
echo.
echo Common options:
echo   -Configuration CUDA
echo   -Platform x64
echo   -CudaVersion 13.3
echo   -CudaToolkitDir C:\path\CUDA\v13.3
echo   -TensorRTDir C:\path\TensorRT
echo   -OpenCVDir C:\path\opencv\install
echo   -NonInteractive
echo   -DryRun
echo.
echo Defaults:
echo   Delegates to cuda\build-cuda.ps1
echo   Configuration=CUDA
echo   Platform=x64
exit /b 0
