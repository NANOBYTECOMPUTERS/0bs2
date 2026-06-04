@echo off
setlocal EnableExtensions

pushd "%~dp0" || exit /b 1

echo ============================================================
echo  0BS BUILDER - Double-click build launcher
echo ============================================================
echo.

where powershell >nul 2>nul
if errorlevel 1 (
    echo [BUILDER] ERROR: powershell was not found in PATH.
    echo.
    pause
    popd
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILDER.ps1" %*
set "BUILDER_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%BUILDER_EXIT_CODE%"=="0" (
    echo [BUILDER] Build complete.
) else (
    echo [BUILDER] Build failed with exit code %BUILDER_EXIT_CODE%.
)
echo.
pause

popd
exit /b %BUILDER_EXIT_CODE%
