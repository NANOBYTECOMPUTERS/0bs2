@echo off
setlocal EnableExtensions

pushd "%~dp0" || exit /b 1

where powershell >nul 2>nul
if errorlevel 1 (
    echo [RUN_BUILDER] ERROR: powershell was not found in PATH.
    echo.
    pause
    popd
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0BUILDER.ps1" %*
set "BUILDER_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%BUILDER_EXIT_CODE%"=="0" (
    echo [RUN_BUILDER] Build complete.
) else (
    echo [RUN_BUILDER] Build failed with exit code %BUILDER_EXIT_CODE%.
)
echo.
pause

popd
exit /b %BUILDER_EXIT_CODE%
