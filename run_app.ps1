# Launch IndicAI-Detect Streamlit Application
$env:Path += ";$env:APPDATA\Python\Python313\Scripts"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m streamlit run app/streamlit_app.py
} elseif (Test-Path "C:\Program Files\Python313\python.exe") {
    & "C:\Program Files\Python313\python.exe" -m streamlit run app/streamlit_app.py
} elseif (Test-Path "$env:APPDATA\Python\Python313\Scripts\streamlit.exe") {
    & "$env:APPDATA\Python\Python313\Scripts\streamlit.exe" run app/streamlit_app.py
} else {
    streamlit run app/streamlit_app.py
}
