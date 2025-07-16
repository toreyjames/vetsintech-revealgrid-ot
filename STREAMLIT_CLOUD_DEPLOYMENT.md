# Streamlit Cloud Deployment Guide for RevealGrid OT

## Quick Deployment Steps

### 1. Repository Setup
Your repository should have these files in the root directory:
- `app.py` - Main application
- `main.py` - Alternative entry point
- `requirements.txt` - Dependencies
- `.streamlit/config.toml` - Configuration

### 2. Streamlit Cloud Deployment

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app:**
   - **Repository**: `toreyjames/revealgrid`
   - **Branch**: `main` (or your default branch)
   - **Main file path**: Try one of these options:
     - `app.py` (primary)
     - `main.py` (alternative)
     - `deploy.py` (backup)
5. **Click "Deploy"**

### 3. If Main File Path Doesn't Work

If Streamlit Cloud says the main file path doesn't exist, try these steps:

#### Option A: Use `main.py`
- Set main file path to: `main.py`
- This is a complete copy of the app with all dependencies

#### Option B: Use `deploy.py`
- Set main file path to: `deploy.py`
- This is a wrapper that imports from `app.py`

#### Option C: Check File Structure
Make sure your files are in the root directory, not in subdirectories.

### 4. Alternative: Use Streamlit Cloud's Auto-Detection

1. **Don't specify a main file path**
2. **Let Streamlit Cloud auto-detect**
3. **It should find `app.py` automatically**

### 5. Troubleshooting

#### If deployment fails:
1. **Check the logs** in Streamlit Cloud
2. **Verify all dependencies** are in `requirements.txt`
3. **Make sure files are in the root directory**
4. **Try the alternative entry points** (`main.py` or `deploy.py`)

#### Common issues:
- **File not found**: Check the file path and location
- **Import errors**: Verify all dependencies in `requirements.txt`
- **Port issues**: Use the `.streamlit/config.toml` configuration

### 6. File Structure Verification

Your repository should look like this:
```
revealgrid/
├── app.py                 # Main application
├── main.py                # Alternative entry point
├── deploy.py              # Deployment wrapper
├── requirements.txt       # Dependencies
├── .streamlit/
│   └── config.toml       # Configuration
├── README.md             # Documentation
└── ... (other files)
```

### 7. Quick Test

Before deploying to Streamlit Cloud, test locally:
```bash
streamlit run app.py
# or
streamlit run main.py
```

### 8. Deployment Checklist

- [ ] All files are in the root directory
- [ ] `requirements.txt` includes all dependencies
- [ ] `app.py` or `main.py` exists and works locally
- [ ] Repository is public (for free Streamlit Cloud)
- [ ] No sensitive data in the repository

### 9. Streamlit Cloud Settings

When configuring in Streamlit Cloud:
- **Repository**: `toreyjames/revealgrid`
- **Branch**: `main`
- **Main file path**: `app.py` (or `main.py` if `app.py` doesn't work)
- **Python version**: 3.9 (auto-detected)

### 10. Success Indicators

When deployment is successful, you should see:
- ✅ "Your app is ready!" message
- ✅ A public URL like `https://your-app-name.streamlit.app`
- ✅ The app loads with the interactive map
- ✅ All features work (plant selection, asset details, etc.)

---

**If you're still having issues, try using `main.py` as the main file path - it's a complete standalone version of the app.** 