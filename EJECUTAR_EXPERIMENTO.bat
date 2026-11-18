@echo off
setlocal
cd /d "%~dp0"

rem Prefer the existing thesis environment on this PC.
set "PYTHON="
if exist "%~dp0..\.venv-swinir\Scripts\python.exe" set "PYTHON=%~dp0..\.venv-swinir\Scripts\python.exe"

rem On another PC, create an isolated environment inside this project.
if not defined PYTHON (
    if not exist ".venv\Scripts\python.exe" (
        where py >nul 2>nul
        if not errorlevel 1 (
            py -3 -m venv .venv
        ) else (
            where python >nul 2>nul
            if errorlevel 1 (
                echo Python is not installed. Install Python 3.10 or newer, then run this file again.
                pause
                exit /b 1
            )
            python -m venv .venv
        )
        if errorlevel 1 goto :error
    )
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
)

"%PYTHON%" -c "import torch, torchvision, numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo Installing the project's Python packages. This can take several minutes the first time.
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :error
    "%PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 goto :error
)

echo Running Feature Squeezing on MNIST. If no checkpoint exists, the script downloads the data and trains for 100 epochs.
"%PYTHON%" ".\src\replicate_mnist_fgsm.py"
if errorlevel 1 goto :error

if exist ".\results\mnist_fgsm_feature_squeezing.png" start "" ".\results\mnist_fgsm_feature_squeezing.png"
echo Results are saved in the results folder.
pause
exit /b 0

:error
echo The experiment stopped with an error. Read the message above and check the README.
pause
exit /b 1
