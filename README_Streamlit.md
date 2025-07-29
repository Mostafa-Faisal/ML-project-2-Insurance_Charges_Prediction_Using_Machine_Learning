# Insurance Charges Prediction - Streamlit Deployment

🏥 **A web application to predict insurance charges using machine learning models**

## 📋 Prerequisites

Before running the Streamlit app, make sure you have completed the following:

1. **Python Environment**: Python 3.8 or higher
2. **Trained Model**: Run the `Insurance_Charges_Prediction.ipynb` notebook first
3. **Dataset**: Download the insurance dataset from Kaggle (optional - notebook creates sample data if not available)

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
Open and run all cells in `Insurance_Charges_Prediction.ipynb` to:
- Load and preprocess the data
- Train three ML models (Linear Regression, Random Forest, XGBoost)
- Save the best model to the `models/` directory

### Step 3: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

**Or use the batch file (Windows):**
```bash
run_app.bat
```

### Step 4: Open in Browser
The app will automatically open in your default browser at `http://localhost:8501`

## 🎯 Features

### Main Functionality
- **Interactive Prediction Interface**: Easy-to-use sliders and dropdowns
- **Real-time Predictions**: Instant insurance charge predictions
- **Risk Assessment**: Categorizes risk level (Low/Medium/High)
- **Feature Importance**: Shows which factors matter most
- **Insights & Tips**: Personalized recommendations

### User Interface
- **Responsive Design**: Works on desktop and mobile
- **Modern Styling**: Clean, professional appearance
- **Visual Feedback**: Color-coded risk levels and BMI categories
- **Interactive Charts**: Feature importance visualization

## 📊 How It Works

### Input Features
1. **Age**: 18-100 years
2. **Gender**: Male or Female
3. **BMI**: Body Mass Index (15-50)
4. **Children**: Number of dependents (0-10)
5. **Smoker**: Yes or No
6. **Region**: Northeast, Northwest, Southeast, Southwest

### Output
- **Predicted Insurance Charges**: Dollar amount
- **Risk Level**: Low, Medium, or High
- **Personalized Insights**: Based on input factors
- **Feature Importance**: What drives the prediction

## 🔧 Technical Details

### Models Supported
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting algorithm

### Data Processing
- One-hot encoding for categorical variables
- Feature scaling and normalization
- Input validation and error handling

### File Structure
```
week 2/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── run_app.bat                   # Windows batch script to run app
├── Insurance_Charges_Prediction.ipynb  # Model training notebook
├── models/                       # Saved model files (created after training)
│   ├── best_insurance_model.pkl
│   ├── feature_columns.pkl
│   └── model_info.pkl
└── README_Streamlit.md          # This file
```

## 🎨 Customization

### Styling
The app uses custom CSS for:
- Modern color scheme
- Responsive layout
- Interactive elements
- Professional appearance

### Adding Features
You can extend the app by:
- Adding more input validation
- Including additional visualizations
- Implementing model comparison
- Adding data export functionality

## 🚨 Troubleshooting

### Common Issues

**1. "Model files not found" error**
- Solution: Run the Jupyter notebook first to train and save the model

**2. "Import error" for packages**
- Solution: Install requirements with `pip install -r requirements.txt`

**3. "Streamlit command not found"**
- Solution: Make sure Streamlit is installed: `pip install streamlit`

**4. Port already in use**
- Solution: Use a different port: `streamlit run streamlit_app.py --server.port 8502`

### Performance Tips
- Close other browser tabs to free up memory
- Use a modern web browser (Chrome, Firefox, Edge)
- Ensure stable internet connection for initial package downloads

## 📈 Model Performance

The app displays real-time model performance metrics:
- **R² Score**: Model accuracy (higher is better)
- **Model Type**: Which algorithm performed best
- **Training Date**: When the model was last trained

## 🎯 Use Cases

### Educational
- Learn about machine learning in insurance
- Understand feature importance
- Explore data relationships

### Professional
- Insurance premium estimation
- Risk assessment tool
- Client consultation aid

### Personal
- Estimate your insurance costs
- Understand risk factors
- Plan for health improvements

## 📝 Notes

### Disclaimer
- Predictions are estimates based on historical data
- Actual insurance charges may vary
- This tool is for educational/demonstration purposes
- Consult insurance professionals for accurate quotes

### Data Privacy
- No personal data is stored
- All processing happens locally
- No data is sent to external servers

## 🔄 Updates and Maintenance

### Updating the Model
1. Run the notebook with new data
2. The app will automatically use the latest saved model
3. No changes to the Streamlit app code required

### Adding New Features
The modular design makes it easy to:
- Add new input fields
- Include additional models
- Enhance visualizations
- Improve user experience

## 🏆 Credits

Built with:
- **Streamlit**: Web app framework
- **scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualizations

---

**Happy Predicting! 🚀**

For questions or issues, please refer to the notebook documentation or contact the development team.
