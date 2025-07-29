# Google Colab Setup Instructions for Insurance Charges Prediction

## 🚀 Running Your Insurance Prediction Model on Google Colab

### Why Use Google Colab?
- ✅ All major ML packages pre-installed
- ✅ Free GPU/TPU access
- ✅ No local setup required
- ✅ Easy sharing and collaboration
- ✅ Automatic Google Drive integration

### 📝 Step-by-Step Instructions:

#### Step 1: Access Google Colab
1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account

#### Step 2: Upload Your Notebook
**Option A: Upload Existing Notebook**
1. Click "Upload" tab
2. Select your `Insurance_Charges_Prediction.ipynb` file
3. Wait for upload to complete

**Option B: Create New Notebook**
1. Click "New notebook"
2. Copy content from your existing notebook

#### Step 3: Install Additional Packages (if needed)
Add this cell at the beginning of your notebook:

```python
# Install XGBoost (only package not pre-installed in Colab)
!pip install xgboost

# Check if all packages are available
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages loaded successfully!")
```

#### Step 4: Upload Dataset
**Option A: Use Sample Data (No upload needed)**
- Your notebook already creates sample data if the CSV isn't found
- Perfect for testing and learning

**Option B: Upload Real Dataset**
1. Download insurance.csv from Kaggle
2. In Colab, click the folder icon (Files) on the left sidebar
3. Click "Upload to session storage"
4. Select your insurance.csv file

#### Step 5: Run All Cells
1. Go to "Runtime" → "Run all"
2. Or run cells one by one with Shift+Enter

#### Step 6: Save Model Files
Since Colab sessions are temporary, save your trained models:

```python
# Download model files to your computer
from google.colab import files

# Save models (run after training)
files.download('models/best_insurance_model.pkl')
files.download('models/feature_columns.pkl')
files.download('models/model_info.pkl')
```

### 🔄 For Streamlit Deployment:

#### Option A: Use Local Files
1. Download the model files from Colab
2. Place them in your local `models/` folder
3. Run Streamlit locally: `streamlit run streamlit_app.py`

#### Option B: Deploy on Streamlit Cloud
1. Upload your code to GitHub
2. Include the trained model files
3. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)

### 📱 Modified Colab Notebook Cells:

Here are the key modifications needed for Colab:

**Cell 1: Package Installation**
```python
# Install XGBoost for Google Colab
!pip install xgboost

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
import pickle
import os
from google.colab import files  # For downloading files

warnings.filterwarnings('ignore')
plt.style.use('default')  # Use default style in Colab
print("✅ All libraries imported successfully!")
```

**Cell for File Management:**
```python
# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')
    print("📁 Created 'models' directory")

# Optional: Upload dataset (or use sample data)
print("📂 You can upload insurance.csv using the file manager on the left")
print("📝 Or the notebook will use sample data for demonstration")
```

**Cell for Downloading Results:**
```python
# Download trained model files
print("💾 Downloading model files...")
try:
    files.download('models/best_insurance_model.pkl')
    files.download('models/feature_columns.pkl') 
    files.download('models/model_info.pkl')
    print("✅ Model files downloaded successfully!")
    print("📝 Place these files in your local 'models/' folder for Streamlit")
except:
    print("❌ Error downloading files. Make sure models are trained first.")
```

### 🎯 Advantages of Colab Approach:

1. **No Local Setup**: Everything runs in the cloud
2. **Free Resources**: Access to GPUs for faster training
3. **Pre-installed Packages**: Most ML libraries already available
4. **Easy Sharing**: Share notebook with collaborators
5. **Automatic Saving**: Work saved to Google Drive
6. **Version Control**: Built-in revision history

### 🔄 Workflow Summary:

1. **Train Model in Colab** → Download model files
2. **Place files locally** → In your `models/` folder  
3. **Run Streamlit locally** → `streamlit run streamlit_app.py`
4. **Deploy to web** → Using Streamlit Cloud or other platforms

This approach gives you the best of both worlds: cloud-based training and local deployment!

### 🆘 Need Help?

If you encounter any issues:
1. Check that XGBoost installed correctly
2. Verify model files are created and downloaded
3. Ensure local Streamlit app can find the model files
4. All packages are compatible with the latest versions

**Happy machine learning! 🚀**
