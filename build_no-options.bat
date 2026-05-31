@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" (
    echo Usage: build_no-options.bat -Backend DML^|CUDA^|WORKER^|ALL
    echo.
    echo Runs existing 0BS build paths without dependency prompts or setup steps.
    exit /b 0
)
if /I "%~1"=="/?" (
    echo Usage: build_no-options.bat -Backend DML^|CUDA^|WORKER^|ALL
    echo.
    echo Runs existing 0BS build paths without dependency prompts or setup steps.
    exit /b 0
)

pushd "%~dp0" || exit /b 1

where powershell >nul 2>nul
if errorlevel 1 (
    echo [build-no-options] ERROR: powershell was not found in PATH.
    goto fail
)

powershell -NoProfile -ExecutionPolicy Bypass -File "build_no-options.ps1" %*
if errorlevel 1 goto fail

popd
exit /b 0

:fail
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if "%BUILD_EXIT_CODE%"=="0" set "BUILD_EXIT_CODE=1"
echo [build-no-options] Failed with exit code %BUILD_EXIT_CODE%.
pause
popd
exit /b %BUILD_EXIT_CODE%
