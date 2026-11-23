@echo off
echo ======================================================================
echo   Launching IndicAI-Detect Streamlit Web Application...
echo ======================================================================

REM Try py launcher
py -m streamlit run app/streamlit_app.py
if %ERRORLEVEL% EQU 0 goto end

REM Try direct python path
"C:\Program Files\Python313\python.exe" -m streamlit run app/streamlit_app.py
if %ERRORLEVEL% EQU 0 goto end

REM Try appdata script
"%APPDATA%\Python\Python313\Scripts\streamlit.exe" run app/streamlit_app.py

:end
pause
