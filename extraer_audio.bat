@echo off
setlocal enabledelayedexpansion
title Extractor de Audio - Transcriptor Suite
color 0B
chcp 65001 >nul

echo ======================================================
echo       EXTRACTOR DE AUDIO - TRANSCRIPTOR SUITE
echo ======================================================
echo.

:: 1. Buscar FFmpeg en el sistema
set "FFMPEG="
where ffmpeg >nul 2>&1 && set "FFMPEG=ffmpeg"

if "!FFMPEG!"=="" (
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
    ) else if exist "%~dp0ffmpeg.exe" (
        set "FFMPEG=%~dp0ffmpeg.exe"
    ) else if exist "C:\ffmpeg\bin\ffmpeg.exe" (
        set "FFMPEG=C:\ffmpeg\bin\ffmpeg.exe"
    )
)

if "!FFMPEG!"=="" (
    color 0C
    echo [ERROR] No se encontro FFmpeg en tu computadora.
    echo.
    echo Para instalarlo en 1 minuto, abre PowerShell y escribe:
    echo     winget install Gyan.FFmpeg
    echo.
    echo O descarga ffmpeg.exe y colocalo en esta misma carpeta.
    echo ======================================================
    pause
    exit /b 1
)

:: 2. Si no arrastro archivos, pedir ruta por teclado
if "%~1"=="" (
    echo Arrastra un archivo de video a esta ventana o escribe su ruta:
    set /p "INPUT_FILE=> "
    set "INPUT_FILE=!INPUT_FILE:"=!"
    if "!INPUT_FILE!"=="" exit /b 0
    call :PROCESAR "!INPUT_FILE!"
    goto :FIN
)

:: 3. Procesar todos los archivos arrastrados (uno o varios a la vez)
:LOOP
if "%~1"=="" goto :FIN
call :PROCESAR "%~1"
shift
goto :LOOP

:PROCESAR
set "ORIGINAL=%~1"
if not exist "!ORIGINAL!" (
    echo [ERROR] No se encuentra el archivo: "!ORIGINAL!"
    echo.
    exit /b 1
)

set "DESTINO=%~dpn1_audio.m4a"

echo ------------------------------------------------------
echo Procesando: "%~nx1"
echo Extrayendo audio en calidad original (Stream Copy)...

"!FFMPEG!" -y -hide_banner -loglevel warning -stats -i "!ORIGINAL!" -vn -c:a copy "!DESTINO!"

if !ERRORLEVEL! neq 0 (
    echo [AVISO] Formato no compatible con copia directa. Codificando a AAC...
    "!FFMPEG!" -y -hide_banner -loglevel warning -stats -i "!ORIGINAL!" -vn -c:a aac -b:a 64k -ac 1 "!DESTINO!"
)

if !ERRORLEVEL! equ 0 (
    echo.
    echo [EXITO] Audio listo para subir:
    echo         "!DESTINO!"
) else (
    echo.
    echo [ERROR] Ocurrio un fallo al procesar "%~nx1".
)
echo ------------------------------------------------------
echo.
exit /b 0

:FIN
echo ======================================================
echo Proceso finalizado.
echo Puedes subir los audios generados a Transcriptor Suite.
echo ======================================================
echo.
pause
