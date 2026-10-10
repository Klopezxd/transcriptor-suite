@echo off
title Transcriptor Suite
chcp 65001 >nul
color 0B

echo ========================================================
echo   Iniciando Transcriptor Suite (Entorno Conda)
echo ========================================================
echo.

:: 1. Detectar Miniconda / Anaconda en rutas estandar
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" transcriptor
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" transcriptor
) else (
    call conda activate transcriptor 2>nul
)

:: 2. Ejecutar CLI (abrira el selector grafico de archivo automaticamente si no se pasa parametro)
python cli.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [AVISO] El proceso finalizo con codigo de error %ERRORLEVEL%.
    pause
)
