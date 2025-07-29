# Insurance Charges Prediction Using Machine Learning - Step by Step Guide

## Project Overview
**Goal**: Predict insurance charges based on client attributes such as age, BMI, smoking status, and other health factors.

**Techniques Used**: Linear Regression, Random Forest, XGBoost
**Dataset**: https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender

## Step-by-Step Implementation Guide

### Step 1: Environment Setup
1. **Install Required Libraries**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
   ```

2. **Import Necessary Libraries**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.ensemble import RandomForestRegressor
   from sklearn.metrics import mean_squared_error, r2_score
   import xgboost as xgb
   import streamlit as st
   ```

### Step 2: Data Loading and Preprocessing
1. **Load the Dataset**
   ```python
   # Download dataset from Kaggle or load from CSV
   df = pd.read_csv('insurance.csv')
   print(df.head())
   print(df.info())
   print(df.describe())
   ```

2. **Data Exploration**
   ```python
   # Check for missing values
   print(df.isnull().sum())
   
   # Check data types
   print(df.dtypes)
   
   # Check unique values for categorical columns
   print(df['sex'].unique())
   print(df['smoker'].unique())
   print(df['region'].unique())
   ```

3. **Data Visualization**
   ```python
   # Distribution plots
   plt.figure(figsize=(15, 10))
   
   # Age distribution
   plt.subplot(2, 3, 1)
   plt.hist(df['age'], bins=20)
   plt.title('Age Distribution')
   
   # BMI distribution
   plt.subplot(2, 3, 2)
   plt.hist(df['bmi'], bins=20)
   plt.title('BMI Distribution')
   
   # Charges distribution
   plt.subplot(2, 3, 3)
   plt.hist(df['charges'], bins=20)
   plt.title('Charges Distribution')
   
   # Smoker vs Charges
   plt.subplot(2, 3, 4)
   sns.boxplot(x='smoker', y='charges', data=df)
   plt.title('Smoker vs Charges')
   
   # Region vs Charges
   plt.subplot(2, 3, 5)
   sns.boxplot(x='region', y='charges', data=df)
   plt.title('Region vs Charges')
   
   # Age vs Charges
   plt.subplot(2, 3, 6)
   plt.scatter(df['age'], df['charges'])
   plt.title('Age vs Charges')
   
   plt.tight_layout()
   plt.show()
   ```

4. **Data Preprocessing**
   ```python
   # Handle categorical variables
   df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
   
   # Check correlation matrix
   plt.figure(figsize=(12, 8))
   correlation_matrix = df_encoded.corr()
   sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
   plt.title('Correlation Matrix')
   plt.show()
   ```

### Step 3: Model Training and Evaluation
1. **Prepare Features and Target**
   ```python
   # Separate features and target
   X = df_encoded.drop('charges', axis=1)
   y = df_encoded['charges']
   
   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   print(f"Training set size: {X_train.shape}")
   print(f"Test set size: {X_test.shape}")
   ```

2. **Model 1: Linear Regression**
   ```python
   # Train Linear Regression
   lr_model = LinearRegression()
   lr_model.fit(X_train, y_train)
   
   # Make predictions
   lr_pred = lr_model.predict(X_test)
   
   # Evaluate
   lr_mse = mean_squared_error(y_test, lr_pred)
   lr_r2 = r2_score(y_test, lr_pred)
   
   print(f"Linear Regression - MSE: {lr_mse:.2f}, R²: {lr_r2:.4f}")
   ```

3. **Model 2: Random Forest**
   ```python
   # Train Random Forest
   rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
   rf_model.fit(X_train, y_train)
   
   # Make predictions
   rf_pred = rf_model.predict(X_test)
   
   # Evaluate
   rf_mse = mean_squared_error(y_test, rf_pred)
   rf_r2 = r2_score(y_test, rf_pred)
   
   print(f"Random Forest - MSE: {rf_mse:.2f}, R²: {rf_r2:.4f}")
   
   # Feature importance
   feature_importance = pd.DataFrame({
       'feature': X.columns,
       'importance': rf_model.feature_importances_
   }).sort_values('importance', ascending=False)
   
   plt.figure(figsize=(10, 6))
   sns.barplot(x='importance', y='feature', data=feature_importance)
   plt.title('Feature Importance (Random Forest)')
   plt.show()
   ```

4. **Model 3: XGBoost**
   ```python
   # Train XGBoost
   xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
   xgb_model.fit(X_train, y_train)
   
   # Make predictions
   xgb_pred = xgb_model.predict(X_test)
   
   # Evaluate
   xgb_mse = mean_squared_error(y_test, xgb_pred)
   xgb_r2 = r2_score(y_test, xgb_pred)
   
   print(f"XGBoost - MSE: {xgb_mse:.2f}, R²: {xgb_r2:.4f}")
   ```

### Step 4: Model Comparison and Visualization
```python
# Compare all models
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MSE': [lr_mse, rf_mse, xgb_mse],
    'R²': [lr_r2, rf_r2, xgb_r2]
})

print("\nModel Comparison:")
print(results)

# Visualize predictions vs actual
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, lr_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Linear Regression (R² = {lr_r2:.4f})')

plt.subplot(1, 3, 2)
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Random Forest (R² = {rf_r2:.4f})')

plt.subplot(1, 3, 3)
plt.scatter(y_test, xgb_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'XGBoost (R² = {xgb_r2:.4f})')

plt.tight_layout()
plt.show()
```

### Step 5: Streamlit Deployment
1. **Create Streamlit App** (save as `app.py`)
   ```python
   import streamlit as st
   import pandas as pd
   import numpy as np
   import pickle
   from sklearn.ensemble import RandomForestRegressor
   
   st.title("Insurance Charges Prediction")
   st.write("Predict insurance charges based on personal attributes")
   
   # Input features
   age = st.slider("Age", 18, 100, 30)
   sex = st.selectbox("Sex", ["male", "female"])
   bmi = st.slider("BMI", 15.0, 50.0, 25.0)
   children = st.slider("Number of Children", 0, 10, 0)
   smoker = st.selectbox("Smoker", ["yes", "no"])
   region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
   
   # Create input dataframe
   input_data = pd.DataFrame({
       'age': [age],
       'sex': [sex],
       'bmi': [bmi],
       'children': [children],
       'smoker': [smoker],
       'region': [region]
   })
   
   # Encode categorical variables
   input_encoded = pd.get_dummies(input_data)
   
   # Make prediction button
   if st.button("Predict Insurance Charges"):
       # Load your trained model (you need to save it first)
       # model = pickle.load(open('insurance_model.pkl', 'rb'))
       # prediction = model.predict(input_encoded)
       # st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")
       st.info("Please train and save your model first!")
   ```

2. **Run Streamlit App**
   ```bash
   streamlit run app.py
   ```

### Step 6: Model Deployment Preparation
1. **Save the Best Model**
   ```python
   import pickle
   
   # Save the best performing model
   with open('insurance_model.pkl', 'wb') as f:
       pickle.dump(rf_model, f)  # or whichever model performed best
   
   # Save the feature columns for consistency
   with open('feature_columns.pkl', 'wb') as f:
       pickle.dump(X.columns.tolist(), f)
   ```

### Key Points to Remember:
- **Data Quality**: Always check for missing values, outliers, and data consistency
- **Feature Engineering**: Consider creating new features like BMI categories, age groups
- **Model Selection**: Choose the model with the best balance of accuracy and interpretability
- **Validation**: Use cross-validation for more robust model evaluation
- **Deployment**: Test your Streamlit app thoroughly before deployment

### Next Steps:
1. Download the dataset from Kaggle
2. Create a Jupyter notebook and follow the steps above
3. Experiment with different hyperparameters
4. Try additional models like Support Vector Regression
5. Deploy your final model using Streamlit

Would you like me to help you create a Jupyter notebook to implement this project?