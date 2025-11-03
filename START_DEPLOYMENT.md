# 🚀 Start/Restart Deployment on Streamlit Cloud

## Method 1: Restart from Dashboard (Recommended)

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account (@ganeshadapnor)
3. **Find your app**: `rag_model · main · rag_streamlit.py`
4. **Click on your app** to open it
5. **Click "Manage app"** (three dots menu or gear icon ⚙️)
6. **Click "Restart app"** or **"Redeploy"**
7. **Wait 2-5 minutes** for rebuild

## Method 2: Delete and Recreate (If Restart Doesn't Work)

1. **Go to**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Find your app**: `rag_model`
4. **Click "Manage app"** → **"Delete app"**
5. **Click "New app"** button
6. **Fill in**:
   - **Repository**: `GaneshAdapnor/RAG_model`
   - **Branch**: `main`
   - **Main file**: `rag_streamlit.py`
   - **App URL**: Choose a name (e.g., `rag-model` or `rag-pdf-qa`)
7. **Click "Deploy"**
8. **Wait 2-5 minutes** for build

## Method 3: Auto-Restart via Git Push

If you make any changes and push to GitHub, Streamlit Cloud will auto-restart:

```bash
git add .
git commit -m "Update deployment"
git push origin main
```

## Current Status

✅ **All fixes applied**:
- `packages.txt` fixed (removed invalid packages)
- `requirements.txt` fixed (removed explicit torch)
- `st.set_page_config` moved to top of file
- Error handling added

✅ **All files pushed to GitHub**

## What to Expect

1. **Build starts** (1-2 minutes to detect changes)
2. **Dependencies install** (2-5 minutes)
3. **App deploys** (1-2 minutes)
4. **Status changes** from "Building" → "Running" ✅

## If Still Red Error

1. **Check logs** for specific error
2. **Share error message** to fix it
3. **Try Method 2** (delete and recreate)

## Your App Details

- **Repository**: https://github.com/GaneshAdapnor/RAG_model.git
- **Main file**: `rag_streamlit.py`
- **Branch**: `main`
- **Streamlit Cloud**: https://share.streamlit.io

## Quick Steps

1. Go to https://share.streamlit.io
2. Click on `rag_model` app
3. Click "Manage app" → "Restart app"
4. Wait 2-5 minutes
5. Check status (should show "Running" ✅)

