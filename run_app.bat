@echo off
echo Starting Insurance Charges Prediction App...
echo.
echo Make sure you have:
echo 1. Installed all required packages (run setup.bat if needed)
echo 2. Trained the model (either locally or in Google Colab)
echo 3. Downloaded model files from Colab to 'models' directory (if using Colab)
echo 4. The 'models' directory exists with trained model files
echo.

REM Check if models directory exists
if not exist "models" (
    echo ❌ Models directory not found!
    echo.
    echo If you used Google Colab:
    echo 1. Download the model files from Colab
    echo 2. Create a 'models' folder here
    echo 3. Place the downloaded .pkl files in the models folder
    echo.
    echo If training locally:
    echo 1. Run the Jupyter notebook first
    echo 2. Make sure all cells execute successfully
    echo.
    pause
    exit /b 1
)

REM Check if model files exist
if not exist "models\best_insurance_model.pkl" (
    echo ❌ Model files not found in models directory!
    echo.
    echo Please ensure you have these files:
    echo • best_insurance_model.pkl
    echo • feature_columns.pkl  
    echo • model_info.pkl
    echo.
    echo Download them from Google Colab or run the training notebook locally.
    echo.
    pause
    exit /b 1
)

echo ✅ Model files found!
echo.
echo Starting Streamlit app...
streamlit run streamlit_app.py
