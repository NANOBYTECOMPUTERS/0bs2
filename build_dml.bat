@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto usage
if /I "%~1"=="/?" goto usage

pushd "%~dp0" || exit /b 1

where powershell >nul 2>nul
if errorlevel 1 (
    echo [build-dml] ERROR: powershell was not found in PATH.
    goto fail
)

powershell -NoProfile -ExecutionPolicy Bypass -File "tools\build_dml.ps1" %*
if errorlevel 1 goto fail

popd
exit /b 0

:fail
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if "%BUILD_EXIT_CODE%"=="0" set "BUILD_EXIT_CODE=1"
echo [build-dml] Failed with exit code %BUILD_EXIT_CODE%.
pause
popd
exit /b %BUILD_EXIT_CODE%

:usage
echo Usage: build_dml.bat [tools\build_dml.ps1 arguments] [extra MSBuild properties]
echo.
echo Common options:
echo   -Configuration DML
echo   -Platform x64
echo   -NonInteractive
echo   -DryRun
echo.
echo Defaults:
echo   Project=0BS_box_2.vcxproj
echo   Configuration=DML
echo   Platform=x64
exit /b 0
