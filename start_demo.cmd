@echo off
cd /d "%~dp0"
title BLCU-QA demo

rem int8: rerank is ~1.7x faster on CPU, ranking differences are inside noise.
set BLCU_RERANK_INT8=1

echo ============================================================
echo  BLCU-QA demo  ^|  http://127.0.0.1:8000
echo  First start loads models and builds the index: about 40 s.
echo  Wait until you see "Uvicorn running on http://127.0.0.1:8000",
echo  the browser opens by itself. Ctrl+C or close this window to stop.
echo  (rerank runs in int8: about 1.1 s per question instead of 1.9 s)
echo ============================================================
echo.

rem open the browser 40 s later, so it does not hit a dead port
start "" powershell -NoProfile -Command "Start-Sleep -Seconds 40; Start-Process 'http://127.0.0.1:8000/'"

python api.py

if errorlevel 1 (
    echo.
    echo [failed] check: python on PATH / pip install -r requirements.txt
    echo          port 8000 busy?  netstat -ano ^| findstr :8000
    pause
)
