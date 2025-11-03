# 🔄 How to Restart Deployment in Streamlit Cloud

## Quick Restart Steps

### Method 1: Restart from Dashboard

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account (@ganeshadapnor)
3. **Find your app** in the dashboard:
   - Look for: `rag-model` or `rag-streamlit-dquy4g`
   - Or search for: `RAG_model`
4. **Click on your app** to open it
5. **Click "Manage app"** (gear icon ⚙️ or three dots menu)
6. **Click "Restart app"** or **"Redeploy"**
7. **Wait 2-5 minutes** for rebuild

### Method 2: Push to GitHub (Auto-restart)

Streamlit Cloud automatically redeploys when you push to GitHub:

```bash
git push origin main
```

The deployment will automatically restart.

### Method 3: Delete and Recreate

If restart doesn't work:

1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Manage app" → "Delete app"
4. Click "New app" and redeploy

## Current Status

✅ **Fix Applied**: `packages.txt` issue fixed
✅ **Pushed to GitHub**: All changes committed
✅ **Auto-restart**: Streamlit Cloud should auto-redeploy

## Your App Details

- **Repository**: https://github.com/GaneshAdapnor/RAG_model.git
- **App URL**: https://ganeshadapnor-rag-model-rag-streamlit-dquy4g.streamlit.app
- **Streamlit Cloud**: https://share.streamlit.io

## After Restart

1. Wait 2-5 minutes for build
2. Check logs in dashboard
3. Visit your app URL
4. App should work now!

## Troubleshooting

If restart doesn't work:
- Check logs for errors
- Verify all files are in GitHub
- Try deleting and recreating the app

