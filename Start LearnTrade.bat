@echo off
REM Windows double-click launcher for LearnTrade
REM If SmartScreen appears: More info -> Run anyway (only needed first time)

setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "LOG_DIR=%SCRIPT_DIR%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\launcher.log"

echo [%date% %time%] Starting LearnTrade... >> "%LOG_FILE%"
echo Starting LearnTrade...

REM Install uv if missing
where uv >nul 2>&1
if errorlevel 1 (
    echo Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Could not install uv. See logs\launcher.log for details.
        pause
        exit /b 1
    )
    REM The uv installer updates the registry PATH for new terminals, but this
    REM already-running process still has the old PATH. Add the two possible
    REM install locations directly so the rest of this script can find it.
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv was installed but is not on PATH in this window.
        echo Close this window, open a NEW Command Prompt, and double-click
        echo this launcher again.
        pause
        exit /b 1
    )
)

echo Checking dependencies...
uv sync --quiet
if errorlevel 1 (
    echo ERROR: Dependency install failed. See logs\launcher.log
    pause
    exit /b 1
)

if "%LEARNTRADE_PORT%"=="" set "LEARNTRADE_PORT=8787"
set "URL=http://localhost:%LEARNTRADE_PORT%"

REM Open the browser only once the server actually answers, instead of
REM guessing a fixed delay. Runs in its own window so it never blocks the
REM server itself.
start "" /min cmd /c ^
    "for /l %%i in (1,1,60) do (curl -fs %URL%/api/health >nul 2>&1 && start "" "%URL%" && exit /b 0 || timeout /t 1 /nobreak >nul)"

echo Starting server at %URL%
uv run learntrade serve --port %LEARNTRADE_PORT% >> "%LOG_FILE%" 2>&1
pause
