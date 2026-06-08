@echo off
title Suno Process Pipeline
cd /d "%~dp0"
echo ================================================
echo Suno Audio Processing Pipeline
echo ================================================
echo.

:: Remove the broken gdk_pixbuf DLL directory from PATH so it doesn't get loaded
set PATH=%PATH:C:\Users\MaxGlasser\.conda\envs\ai0a\Library\bin;=%

if "%~1"=="" (
    echo Drag and drop an MP3 onto this .bat file to process it.
    echo Or run: suno-process.bat "path\to\file.mp3"
    pause
    exit /b
)

"C:\Users\MaxGlasser\.conda\envs\ai0a\python.exe" suno-process.py %*
echo.
pause
