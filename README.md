# 🏥 Insurance Charges Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green.svg)](https://xgboost.readthedocs.io/)

## 📋 Project Overview

This project predicts medical insurance charges using machine learning algorithms. The application analyzes various personal and medical factors to estimate insurance costs, helping users understand how different factors influence their insurance premiums.

## 🎯 Objectives

- Predict insurance charges based on personal and medical information
- Compare performance of different machine learning algorithms
- Provide an interactive web application for real-time predictions
- Analyze feature importance to understand cost drivers

## 📊 Dataset

The project uses a medical insurance dataset containing the following features:

- **Age**: Age of the primary beneficiary
- **Sex**: Insurance contractor gender (female/male)
- **BMI**: Body mass index (kg/m²)
- **Children**: Number of children covered by health insurance
- **Smoker**: Smoking status (yes/no)
- **Region**: Residential area in the US (northeast, southeast, southwest, northwest)
- **Charges**: Individual medical costs billed by health insurance (target variable)

## 🚀 Features

- **Interactive Web App**: User-friendly Streamlit interface
- **Multiple ML Models**: Comparison of different algorithms
- **Real-time Predictions**: Instant insurance charge estimates
- **Data Visualization**: Charts and graphs for better insights
- **Feature Importance**: Understanding which factors matter most
- **Model Performance Metrics**: Comprehensive evaluation results

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization

## 📁 Project Structure

```
├── Insurance_Charges_Prediction_Using_Machine_Learning.ipynb  # Main analysis notebook
├── Insurance_Charges_Prediction.ipynb                        # Alternative notebook
├── Insurance_Charges_Prediction_Colab.ipynb                 # Google Colab version
├── streamlit_app.py                                          # Web application
├── insurance.csv                                             # Dataset
├── requirements.txt                                          # Python dependencies
├── models/                                                   # Trained models
│   ├── best_insurance_model.pkl
│   ├── feature_columns.pkl
│   └── model_info.pkl
├── submit/                                                   # Submission files
├── setup.bat                                                # Windows setup script
├── run_app.bat                                              # Windows run script
├── PYTHON_SETUP_GUIDE.md                                    # Python setup guide
├── GOOGLE_COLAB_GUIDE.md                                    # Colab setup guide
└── README_Streamlit.md                                      # Streamlit specific docs
```

## 🔧 Installation & Setup

### Method 1: Automatic Setup (Windows)
```bash
# Clone the repository
git clone https://github.com/Mostafa-Faisal/ML-project-2-Insurance_Charges_Prediction_Using_Machine_Learning.git
cd ML-project-2-Insurance_Charges_Prediction_Using_Machine_Learning

# Run setup script (Windows)
setup.bat

# Run the application
run_app.bat
```

### Method 2: Manual Setup
```bash
# Clone the repository
git clone https://github.com/Mostafa-Faisal/ML-project-2-Insurance_Charges_Prediction_Using_Machine_Learning.git
cd ML-project-2-Insurance_Charges_Prediction_Using_Machine_Learning

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Method 3: Google Colab
Open `Insurance_Charges_Prediction_Colab.ipynb` in Google Colab for a cloud-based solution.

## 💻 Usage

1. **Launch the Web App**: Run `streamlit run streamlit_app.py`
2. **Enter Details**: Input personal information in the sidebar
3. **Get Prediction**: View estimated insurance charges
4. **Explore Models**: Compare different algorithm performances
5. **Analyze Results**: Review feature importance and model metrics

## 📈 Model Performance

The project implements and compares multiple machine learning algorithms:

- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Support Vector Regression**
- **Gradient Boosting Regressor**

Performance metrics include:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## 🔍 Key Insights

- Smoking status is the most significant factor affecting insurance charges
- Age and BMI show strong positive correlation with charges
- Number of children and gender have minimal impact
- Regional differences exist but are less pronounced

## 📱 Web Application Features

- **User Input Panel**: Easy-to-use sidebar for data entry
- **Prediction Display**: Clear visualization of predicted charges
- **Model Comparison**: Side-by-side algorithm performance
- **Interactive Charts**: Dynamic data visualization
- **Export Options**: Download predictions and reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mostafa Faisal**
- GitHub: [@Mostafa-Faisal](https://github.com/Mostafa-Faisal)
- Email: mostafa.faisal@example.com

## 🙏 Acknowledgments

- Dataset source: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspiration from various Kaggle kernels and ML tutorials
- Thanks to the open-source community for the amazing tools

## 📞 Support

If you have any questions or need help, please:
1. Check the documentation files in the repository
2. Open an issue on GitHub
3. Contact the author directly

---

⭐ **Star this repository if you found it helpful!** ⭐
