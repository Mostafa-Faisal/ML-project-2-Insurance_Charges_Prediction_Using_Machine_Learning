@echo off
echo ========================================
echo   Insurance ML Project Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo Please install Python first:
    echo 1. Go to https://www.python.org/downloads/
    echo 2. Download Python 3.9 or higher
    echo 3. During installation, CHECK "Add Python to PATH"
    echo 4. Restart this script after installation
    echo.
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

echo Installing required packages...
echo This may take a few minutes...
echo.

echo Installing streamlit...
python -m pip install streamlit
if %errorlevel% neq 0 (
    echo ❌ Failed to install streamlit
    goto :error
)

echo Installing pandas...
python -m pip install pandas
if %errorlevel% neq 0 (
    echo ❌ Failed to install pandas
    goto :error
)

echo Installing numpy...
python -m pip install numpy
if %errorlevel% neq 0 (
    echo ❌ Failed to install numpy
    goto :error
)

echo Installing scikit-learn...
python -m pip install scikit-learn
if %errorlevel% neq 0 (
    echo ❌ Failed to install scikit-learn
    goto :error
)

echo Installing xgboost...
python -m pip install xgboost
if %errorlevel% neq 0 (
    echo ❌ Failed to install xgboost
    goto :error
)

echo Installing matplotlib...
python -m pip install matplotlib
if %errorlevel% neq 0 (
    echo ❌ Failed to install matplotlib
    goto :error
)

echo Installing seaborn...
python -m pip install seaborn
if %errorlevel% neq 0 (
    echo ❌ Failed to install seaborn
    goto :error
)

echo.
echo ✅ All packages installed successfully!
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run the Jupyter notebook to train your model
echo 2. Then run: streamlit run streamlit_app.py
echo.
echo Or just run: run_app.bat
echo.
pause
exit /b 0

:error
echo.
echo ❌ Error installing packages!
echo.
echo Try these solutions:
echo 1. Run this script as Administrator
echo 2. Use: pip install --user [package-name]
echo 3. Install Anaconda and use Anaconda Prompt
echo.
pause
exit /b 1
