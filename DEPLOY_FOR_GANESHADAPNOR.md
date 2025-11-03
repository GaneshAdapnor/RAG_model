# 🚀 Deployment Instructions for @ganeshadapnor

## Your Deployment Details

- **GitHub Repository**: https://github.com/GaneshAdapnor/RAG_model.git
- **Streamlit ID**: @ganeshadapnor
- **Main File**: `rag_streamlit.py`
- **Branch**: `main`

## Step-by-Step Deployment

### 1. Go to Streamlit Cloud
Visit: **https://share.streamlit.io**
- Sign in with your GitHub account (ganeshadapnor)

### 2. Create New App
- Click **"New app"** button
- Or go to: https://share.streamlit.io/signup

### 3. Fill in Deployment Form

**Repository**: 
- Select: `GaneshAdapnor/RAG_model`
- Or search: `RAG_model`

**Branch**: 
- `main`

**Main file path**: 
- `rag_streamlit.py`

**App URL**: 
- Choose a unique name (e.g., `rag-pdf-qa`, `rag-documents`, `pdf-qa-system`)

### 4. Advanced Settings (Optional)

Click **"Advanced settings"** → **"Secrets"**

Add your OpenAI API key (optional):
```
OPENAI_API_KEY = "your-api-key-here"
```

**Note**: If you don't add this, the app will use SentenceTransformers (free, local embeddings)

### 5. Deploy!

Click **"Deploy"** button

### 6. Wait for Build

- Build time: 2-5 minutes
- First deployment: 5-7 minutes (downloads SentenceTransformers model)
- You can watch progress in the dashboard

### 7. Your App is Live!

Your app will be available at:
```
https://YOUR-APP-NAME.streamlit.app
```

## Quick Link

**Deploy Now**: https://share.streamlit.io

## What's Deployed

✅ `rag_streamlit.py` - Main Streamlit app  
✅ `requirements.txt` - All dependencies  
✅ `.streamlit/config.toml` - Configuration  
✅ Error handling for invalid API keys  
✅ Automatic fallback to SentenceTransformers  

## Troubleshooting

If you see a white screen:
1. Check **"Manage app"** → **"Logs"** in Streamlit Cloud
2. Look for error messages
3. Wait 2-5 minutes for first build
4. Hard refresh browser (Ctrl+F5)

## After Deployment

- ✅ Share your app URL with anyone
- ✅ Auto-updates when you push to GitHub
- ✅ Check logs in dashboard if needed

## Your Repository

All files are ready at: https://github.com/GaneshAdapnor/RAG_model.git

Happy deploying! 🚀

