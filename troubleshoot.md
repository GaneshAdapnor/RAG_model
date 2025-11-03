# Troubleshooting White Screen on Streamlit Cloud

## What a White Screen Means

A white screen usually indicates:
1. **App is still loading** - Wait a few more seconds
2. **Import error** - Check build logs
3. **Runtime error** - Check application logs
4. **App crashed** - Check logs for errors

## How to Check Logs

### In Streamlit Cloud Dashboard:

1. Go to your app dashboard: https://share.streamlit.io
2. Click on your app
3. Click **"Manage app"** → **"Logs"**
4. Look for error messages

### Common Issues and Fixes

#### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'X'`
**Fix**: Add missing package to `requirements.txt`

#### Issue 2: Memory Issues
**Error**: Out of memory
**Fix**: SentenceTransformers uses ~500MB. Consider using OpenAI embeddings instead.

#### Issue 3: Page Config Error
**Error**: `StreamlitAPIException: set_page_config must be called first`
**Fix**: Already fixed in code - page config is wrapped in try/except

#### Issue 4: Cache Directory Error
**Error**: Permission denied creating cache
**Fix**: Already handled - cache creation is optional

## Quick Checks

1. **Check build status**: Is the app still building?
2. **Check logs**: Look for error messages
3. **Check browser console**: Press F12 → Console tab
4. **Hard refresh**: Ctrl+F5 to clear cache

## Test Locally First

Run locally to test:
```bash
streamlit run rag_streamlit.py
```

If it works locally but not on Streamlit Cloud, check:
- All dependencies in `requirements.txt`
- File paths are correct
- No local file system dependencies

## Still Not Working?

1. Check Streamlit Cloud logs
2. Share the error message
3. Verify all files are in GitHub repository

