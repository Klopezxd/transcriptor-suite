@echo off
setlocal enabledelayedexpansion
title Extractor de Audio - Transcriptor Suite
color 0B
chcp 65001 >nul

echo ======================================================
echo       EXTRACTOR DE AUDIO - TRANSCRIPTOR SUITE
echo ======================================================
echo.

:: 1. Buscador inteligente de FFmpeg (Deteccion multinivel con prioridad portable)
set "FFMPEG="

:: A. Prioridad 1: Binario portable local (Opcion A - Baterias incluidas)
if exist "%~dp0bin\ffmpeg.exe" (
    set "FFMPEG=%~dp0bin\ffmpeg.exe"
) else if exist "%~dp0ffmpeg.exe" (
    set "FFMPEG=%~dp0ffmpeg.exe"
)

:: B. Prioridad 2: PATH del sistema
if "!FFMPEG!"=="" (
    where ffmpeg >nul 2>&1 && set "FFMPEG=ffmpeg"
)

:: C. Prioridad 3: WinGet (Links o busqueda profunda en paquetes instalados)
if "!FFMPEG!"=="" (
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" (
        set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
    ) else (
        for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\*ffmpeg*") do (
            for /r "%%D" %%F in (ffmpeg.exe) do (
                if exist "%%F" if "!FFMPEG!"=="" set "FFMPEG=%%F"
            )
        )
    )
)

:: D. Prioridad 4: Entorno Conda del proyecto transcriptor si existe
if "!FFMPEG!"=="" (
    if exist "%USERPROFILE%\miniconda3\envs\transcriptor\Library\bin\ffmpeg.exe" (
        set "FFMPEG=%USERPROFILE%\miniconda3\envs\transcriptor\Library\bin\ffmpeg.exe"
    ) else if exist "%USERPROFILE%\anaconda3\envs\transcriptor\Library\bin\ffmpeg.exe" (
        set "FFMPEG=%USERPROFILE%\anaconda3\envs\transcriptor\Library\bin\ffmpeg.exe"
    )
)

:: E. Prioridad 5: Ubicaciones estandar (C:\ffmpeg, Scoop, Chocolatey, AppData)
if "!FFMPEG!"=="" (
    if exist "%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe" (
        set "FFMPEG=%LOCALAPPDATA%\ffmpeg\bin\ffmpeg.exe"
    ) else if exist "C:\ffmpeg\bin\ffmpeg.exe" (
        set "FFMPEG=C:\ffmpeg\bin\ffmpeg.exe"
    ) else if exist "%USERPROFILE%\scoop\shims\ffmpeg.exe" (
        set "FFMPEG=%USERPROFILE%\scoop\shims\ffmpeg.exe"
    ) else if exist "C:\ProgramData\chocolatey\bin\ffmpeg.exe" (
        set "FFMPEG=C:\ProgramData\chocolatey\bin\ffmpeg.exe"
    )
)

:: 2. Si no se encontro, ofrecer autoinstalacion portable en 1 clic
if "!FFMPEG!"=="" (
    color 0E
    echo [AVISO] FFmpeg no esta disponible en este equipo.
    echo FFmpeg es imprescindible para procesar y optimizar el audio.
    echo.
    set /p "INSTALAR=¿Deseas descargarlo e instalarlo automaticamente ahora? (S/N): "
    if /i "!INSTALAR!"=="S" (
        echo.
        echo [1/2] Descargando FFmpeg portable para la suite...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "$ProgressPreference = 'SilentlyContinue'; " ^
            "$dest = Join-Path '%~dp0' 'bin'; " ^
            "New-Item -ItemType Directory -Force -Path $dest | Out-Null; " ^
            "$zip = Join-Path $env:TEMP 'ffmpeg.zip'; " ^
            "$ext = Join-Path $env:TEMP 'ffmpeg_ext'; " ^
            "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile $zip; " ^
            "Expand-Archive -Path $zip -DestinationPath $ext -Force; " ^
            "$bin = Get-ChildItem $ext -Recurse -Filter 'ffmpeg.exe' | Select-Object -First 1; " ^
            "if ($bin) { Copy-Item $bin.FullName (Join-Path $dest 'ffmpeg.exe') -Force }; " ^
            "Remove-Item $zip -Force -ErrorAction SilentlyContinue; " ^
            "Remove-Item $ext -Recurse -Force -ErrorAction SilentlyContinue"
        
        if exist "%~dp0bin\ffmpeg.exe" (
            set "FFMPEG=%~dp0bin\ffmpeg.exe"
        )
        
        :: Si fallo PowerShell, intentar via WinGet como alternativa
        if "!FFMPEG!"=="" (
            echo [2/2] Intentando via WinGet...
            winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
            for /d %%D in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages\*ffmpeg*") do (
                for /r "%%D" %%F in (ffmpeg.exe) do (
                    if exist "%%F" if "!FFMPEG!"=="" set "FFMPEG=%%F"
                )
            )
            if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
        )
    )
)

:: 3. Verificacion final
if "!FFMPEG!"=="" (
    color 0C
    echo.
    echo ======================================================
    echo [ERROR] No se pudo obtener FFmpeg automaticamente.
    echo.
    echo Solucion manual:
    echo 1. Descarga ffmpeg.exe desde: https://www.gyan.dev/ffmpeg/builds/
    echo 2. Coloca ffmpeg.exe dentro de la carpeta:
    echo    "%~dp0bin\ffmpeg.exe"
    echo ======================================================
    pause
    exit /b 1
)

color 0B

:: 4. Si no arrastro archivos, pedir ruta por teclado
if "%~1"=="" (
    echo Arrastra un archivo de video a esta ventana o escribe su ruta:
    set /p "INPUT_FILE=> "
    set "INPUT_FILE=!INPUT_FILE:"=!"
    if "!INPUT_FILE!"=="" exit /b 0
    call :PROCESAR "!INPUT_FILE!"
    goto :FIN
)

:: 5. Procesar todos los archivos arrastrados (uno o varios a la vez)
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