@echo off
SET VENV_NAME=env

REM Check if venv folder exists
IF NOT EXIST %VENV_NAME% (
    echo Creating virtual environment...
    python -m venv %VENV_NAME%
)

REM Activate the virtual environment
call %VENV_NAME%\Scripts\activate

REM Install requirements (only first time)
pip install -r requirements.txt

REM Run the dashboard
echo Launching the MEECC Dashboard...
python MEECC_V.1_revision.py

pause
