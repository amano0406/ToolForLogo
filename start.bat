@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "DOCKER_DESKTOP_URL=https://docs.docker.com/desktop/setup/install/windows-install/"

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is not installed or docker.exe is not on PATH.
  echo   %DOCKER_DESKTOP_URL%
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop is installed but the Docker engine is not ready.
  echo   %DOCKER_DESKTOP_URL%
  exit /b 1
)

if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Created .env from .env.example.
)

set "WEB_PORT=19130"
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /I "%%A"=="TOOL_FOR_LOGO_WEB_PORT" set "WEB_PORT=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_APPDATA_ROOT" set "HOST_APPDATA=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_UPLOADS_ROOT" set "HOST_UPLOADS=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_OUTPUTS_ROOT" set "HOST_OUTPUTS=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_REPORT_ROOT" set "HOST_REPORTS=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_ARCHIVE_ROOT" set "HOST_ARCHIVE=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_HF_CACHE_ROOT" set "HOST_HF_CACHE=%%B"
  if /I "%%A"=="TOOL_FOR_LOGO_HOST_TORCH_CACHE_ROOT" set "HOST_TORCH_CACHE=%%B"
)

for %%P in ("!HOST_APPDATA!" "!HOST_UPLOADS!" "!HOST_OUTPUTS!" "!HOST_REPORTS!" "!HOST_ARCHIVE!" "!HOST_HF_CACHE!" "!HOST_TORCH_CACHE!") do (
  if not "%%~P"=="" mkdir "%%~P" >nul 2>&1
)

echo Starting ToolForLogo web and worker containers...
docker compose up --build -d
if errorlevel 1 (
  echo docker compose failed before the app became ready.
  exit /b 1
)

echo Waiting for containers and web health check...
set /a ATTEMPT=0

:wait_loop
set /a ATTEMPT+=1
set "WEB_RUNNING="
set "WORKER_RUNNING="
for /f %%S in ('docker compose ps --services --status running 2^>nul') do (
  if /I "%%S"=="web" set "WEB_RUNNING=1"
  if /I "%%S"=="worker" set "WORKER_RUNNING=1"
)

if defined WEB_RUNNING if defined WORKER_RUNNING (
  powershell -NoLogo -NoProfile -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:%WEB_PORT%/health' -UseBasicParsing -TimeoutSec 5; if ($response.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 goto ready
)

if !ATTEMPT! GEQ 60 goto failed
powershell -NoLogo -NoProfile -Command "Start-Sleep -Seconds 2" >nul 2>&1
goto wait_loop

:ready
echo ToolForLogo is ready at http://localhost:%WEB_PORT%
if /I "%TOOL_FOR_LOGO_SKIP_BROWSER_OPEN%"=="1" exit /b 0
start "" "http://localhost:%WEB_PORT%"
exit /b 0

:failed
echo ToolForLogo did not become ready in time.
echo.
docker compose ps
echo.
echo Last container logs:
docker compose logs --tail 80 web worker
exit /b 1
