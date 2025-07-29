# Python Installation and Setup Guide

## 🐍 Python Installation Issue Fix

The error you're encountering indicates that Python is not properly installed or configured on your system.

## 🔧 Solution Options:

### Option 1: Install Python from Official Website (Recommended)

1. **Download Python:**
   - Go to https://www.python.org/downloads/
   - Download Python 3.9 or higher (3.11 recommended)
   - Choose "Add Python to PATH" during installation ✅

2. **Installation Steps:**
   ```
   1. Run the downloaded installer
   2. ✅ CHECK "Add Python to PATH" (VERY IMPORTANT!)
   3. Click "Install Now"
   4. Wait for installation to complete
   5. Restart your terminal/command prompt
   ```

### Option 2: Install via Microsoft Store (Alternative)

1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Install the official Python package
4. Restart your terminal

### Option 3: Use Anaconda/Miniconda (Data Science Focused)

1. **Download Anaconda:**
   - Go to https://www.anaconda.com/products/distribution
   - Download and install Anaconda
   - This includes Python + many data science packages

2. **After Installation:**
   - Open "Anaconda Prompt" instead of regular terminal
   - Navigate to your project folder
   - Run the pip install command

## 🧪 Test Your Installation:

After installing Python, test it:
```bash
python --version
# Should show: Python 3.x.x

pip --version
# Should show: pip version info
```

## 📦 Install Project Dependencies:

Once Python is working:
```bash
# Navigate to your project folder
cd "e:\3. Machine learning Internship\week 2"

# Install requirements
pip install -r requirements.txt

# Alternative if pip doesn't work:
python -m pip install -r requirements.txt
```

## 🚨 Common Issues and Fixes:

### Issue 1: "Python not found"
**Fix:** Make sure "Add to PATH" was checked during installation

### Issue 2: "pip not found"
**Fix:** Use `python -m pip` instead of just `pip`

### Issue 3: Permission errors
**Fix:** Run terminal as Administrator, or use `--user` flag:
```bash
pip install --user -r requirements.txt
```

### Issue 4: Still getting errors
**Fix:** Try installing packages individually:
```bash
pip install streamlit
pip install pandas
pip install numpy
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
```

## 🔄 Alternative: Create Virtual Environment (Recommended for Projects)

```bash
# Create virtual environment
python -m venv insurance_env

# Activate it (Windows)
insurance_env\Scripts\activate

# Install packages
pip install -r requirements.txt

# When done, deactivate
deactivate
```

## 🎯 Quick Start Commands (After Python Installation):

```bash
# Check everything is working
python --version
pip --version

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## 📞 Need Help?

If you're still having issues:
1. Make sure you restart your terminal after Python installation
2. Try using "Anaconda Prompt" if you installed Anaconda
3. Run commands as Administrator
4. Check that Python is in your system PATH

---

**Once Python is installed, you'll be able to run your machine learning project! 🚀**
