# 🚀 Deploy to Streamlit Cloud - Quick Guide

## ✅ Your App is Ready!

Your Streamlit app (`rag_streamlit.py`) is ready for deployment! It uses OpenRouter for answering and summarizing, and local SentenceTransformers for embeddings.

## 📋 Deployment Steps

### Step 1: Push to GitHub (if not already done)

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "RAG PDF Q&A System ready for Streamlit Cloud"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"** button
4. **Fill in the form**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `rag_streamlit.py`
   - **App URL**: Choose a unique name (e.g., `rag-pdf-qa`)
5. **Click "Advanced settings"** → **"Secrets"**:
   ```
   OPENROUTER_API_KEYS = "key-one,key-two,key-three"
   ```
   (a single key works too: `OPENROUTER_API_KEY = "your-key"`)
   **Get your free API key**: https://openrouter.ai/settings/keys
6. **Click "Deploy"**

### Step 3: Wait for Deployment

- Build takes 2-5 minutes
- You'll see progress in the dashboard
- Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

## 📁 Required Files (Already in Place)

✅ `rag_streamlit.py` - Main Streamlit app  
✅ `requirements.txt` - All dependencies  
✅ `.streamlit/config.toml` - Streamlit configuration  

## ⚙️ Required: OpenRouter API Key

You need to add your OpenRouter API key in Streamlit Cloud → Secrets:

```
OPENROUTER_API_KEYS = "key-one,key-two,key-three"
```

**Get your free API key**: https://openrouter.ai/settings/keys

**Note**:
- The app uses SentenceTransformers (free, local) for document embeddings
- OpenRouter is required for answering questions and summarizing documents
- Multiple comma-separated keys enable automatic failover if one hits a rate limit

## 🔄 Auto-Updates

Whenever you push to GitHub, Streamlit Cloud automatically redeploys your app!

```bash
git add .
git commit -m "Update"
git push
```

## 📚 Features

- ✅ Multi-format document upload (PDF, Word, TXT, Markdown, HTML, CSV, RTF)
- ✅ OpenRouter API for document answering and summarization
- ✅ SentenceTransformers for embeddings (free, local)
- ✅ Source attribution and chat history
- ✅ Per-visitor session isolation
- ✅ Automatic key failover on rate limits

## 🎯 Quick Access

**Start deployment**: https://share.streamlit.io

Your app will be live in minutes! 🚀
