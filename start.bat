@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "DOCKER_DESKTOP_URL=https://docs.docker.com/desktop/setup/install/windows-install/"

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not installed or docker.exe is not on PATH.
  echo Download and install Docker Desktop here:
  echo   %DOCKER_DESKTOP_URL%
  if /I not "%TOOL_FOR_LOGO_SKIP_HELP_LINK%"=="1" start "" "%DOCKER_DESKTOP_URL%" >nul 2>&1
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is installed but the Docker engine is not ready.
  echo Start Docker Desktop, wait until the engine is running, and try again.
  if /I not "%TOOL_FOR_LOGO_SKIP_HELP_LINK%"=="1" start "" "%DOCKER_DESKTOP_URL%" >nul 2>&1
  exit /b 1
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example.
)

set "WEB_PORT=19130"
set "HOST_STATE_ROOT=C:\Codex\workspaces\ToolForLogo"
set "HOST_REPORT_ROOT=C:\Codex\reports\ToolForLogo"
set "HOST_ARCHIVE_ROOT=C:\Codex\archive\ToolForLogo"
set "START_COMFYUI=1"
set "COMFYUI_DIR=C:\apps\ComfyUI_windows_portable"

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /I "%%A"=="TOOL_FOR_LOGO_WEB_PORT" set "WEB_PORT=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_STATE_ROOT" set "HOST_STATE_ROOT=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_REPORT_ROOT" set "HOST_REPORT_ROOT=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_ARCHIVE_ROOT" set "HOST_ARCHIVE_ROOT=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_START_COMFYUI" set "START_COMFYUI=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_COMFYUI_DIR" set "COMFYUI_DIR=%%B"
)

if not exist "%HOST_STATE_ROOT%" mkdir "%HOST_STATE_ROOT%" >nul 2>&1
if not exist "%HOST_REPORT_ROOT%" mkdir "%HOST_REPORT_ROOT%" >nul 2>&1
if not exist "%HOST_ARCHIVE_ROOT%" mkdir "%HOST_ARCHIVE_ROOT%" >nul 2>&1

if /I "%START_COMFYUI%"=="1" (
  call :ensure_comfyui
  if errorlevel 1 exit /b 1
)

echo Starting ToolForLogo web container...
docker compose up --build -d
if errorlevel 1 (
  echo docker compose failed before the app became ready.
  exit /b 1
)

echo Waiting for web health check...
set /a ATTEMPT=0

:wait_loop
set /a ATTEMPT+=1
set "WEB_RUNNING="

for /f %%S in ('docker compose ps --services --status running 2^>nul') do (
  if /I "%%S"=="web" set "WEB_RUNNING=1"
)

if defined WEB_RUNNING (
  powershell -NoLogo -NoProfile -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:%WEB_PORT%/health' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 goto ready
)

if !ATTEMPT! GEQ 45 goto failed

powershell -NoLogo -NoProfile -Command "Start-Sleep -Seconds 2" >nul 2>&1
goto wait_loop

:ready
echo ToolForLogo is ready at http://localhost:%WEB_PORT%
echo Example status endpoint: http://localhost:%WEB_PORT%/api/status
exit /b 0

:failed
echo ToolForLogo did not become ready in time.
echo.
docker compose ps
echo.
echo Last container logs:
docker compose logs --tail 60 web
exit /b 1

:ensure_comfyui
powershell -NoLogo -NoProfile -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8188/system_stats' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
  echo ComfyUI API is already ready at http://127.0.0.1:8188
  exit /b 0
)

if not exist "%COMFYUI_DIR%\run_nvidia_gpu_api.bat" (
  echo ComfyUI was requested but %COMFYUI_DIR%\run_nvidia_gpu_api.bat was not found.
  exit /b 1
)

echo Starting local ComfyUI backend...
start "" /min cmd /c "cd /d \"%COMFYUI_DIR%\" && call run_nvidia_gpu_api.bat"
echo Waiting for ComfyUI API...
set /a COMFY_ATTEMPT=0

:wait_comfy
set /a COMFY_ATTEMPT+=1
powershell -NoLogo -NoProfile -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8188/system_stats' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
  echo ComfyUI API is ready at http://127.0.0.1:8188
  exit /b 0
)

if !COMFY_ATTEMPT! GEQ 60 (
  echo ComfyUI did not become ready in time.
  exit /b 1
)

powershell -NoLogo -NoProfile -Command "Start-Sleep -Seconds 2" >nul 2>&1
goto wait_comfy
