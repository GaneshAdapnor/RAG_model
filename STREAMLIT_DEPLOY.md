# Quick Deploy to Streamlit Cloud

## 🚀 Fast Deployment Steps

### 1. Push to GitHub

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "RAG PDF Q&A System ready for deployment"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Fill in:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file**: `rag_streamlit.py`
   - **App URL**: Choose a name
4. Click **"Advanced settings"** → **"Secrets"**
5. Add (optional):
   ```
   OPENAI_API_KEY = "your-key-here"
   ```
6. Click **"Deploy"**

### 3. Done! 🎉

Your app will be live at: `https://your-app-name.streamlit.app`

## 📋 Required Files

✅ `rag_streamlit.py` - Main app  
✅ `requirements.txt` - Dependencies  
✅ `.streamlit/config.toml` - Config  
✅ `.gitignore` - Git ignore  

## ⚙️ Optional: Add API Key

In Streamlit Cloud → Secrets:
```
OPENAI_API_KEY = "sk-..."
```

**Note**: Without API key, app uses SentenceTransformers (free, local)

## 🔄 Updates

Just push to GitHub - Streamlit Cloud auto-deploys!

```bash
git add .
git commit -m "Update"
git push
```

## 📚 Full Guide

See `DEPLOYMENT.md` for detailed instructions.

