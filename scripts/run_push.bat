@echo off
chcp 65001 >nul
title PaperLens 自动推送
rem PaperLens scheduled push entrypoint (called by Task Scheduler).
rem No credential is stored here; the .env file is read by Python at runtime.
setlocal

rem Resolve the checkout that contains this scripts\ directory. A task
rem registered from a worktree must run that worktree's code and config,
rem never a fixed main-checkout path.
for %%I in ("%~dp0..") do set "PAPERLENS_PROJECT=%%~fI"
if not defined PAPERLENS_PYTHON set "PAPERLENS_PYTHON=python"

if not exist "%PAPERLENS_PYTHON%" (
    where "%PAPERLENS_PYTHON%" >nul 2>nul
    if errorlevel 1 (
        echo ERROR: PaperLens Python interpreter not found: %PAPERLENS_PYTHON% 1>&2
        echo Set PAPERLENS_PYTHON to your project Python, or edit scripts\run_push.bat. 1>&2
        exit /b 1
    )
)
if not exist "%PAPERLENS_PROJECT%/push_run.py" (
    echo ERROR: push_run.py not found under %PAPERLENS_PROJECT% 1>&2
    exit /b 1
)

cd /d "%PAPERLENS_PROJECT%"
if errorlevel 1 (
    echo ERROR: cannot enter project directory %PAPERLENS_PROJECT% 1>&2
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "PAPERLENS_PROGRESS=1"
echo [PaperLens] 正在启动定时推送……
echo [PaperLens] 项目目录: %PAPERLENS_PROJECT%
echo [PaperLens] 正在读取配置、检查邮箱并抓取论文，请稍候……
"%PAPERLENS_PYTHON%" -X utf8 -B "%PAPERLENS_PROJECT%/push_run.py" --scheduled %*
set "PAPERLENS_RC=%ERRORLEVEL%"
echo.
echo [PaperLens] 运行结束，退出码 %PAPERLENS_RC%。
if not defined PAPERLENS_NO_HOLD (
    echo [PaperLens] 窗口将在 6 秒后自动关闭，可查看上面的进度。
    timeout /t 6 /nobreak >nul 2>&1
)
exit /b %PAPERLENS_RC%
