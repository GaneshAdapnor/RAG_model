# 🚀 Deploy to Streamlit Cloud - Quick Guide

## ✅ Your App is Ready!

Your Streamlit app (`rag_streamlit.py`) is ready for deployment!

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
   GOOGLE_API_KEY = "your-gemini-api-key-here"
   ```
   **Get your free API key**: https://aistudio.google.com/apikey
6. **Click "Deploy"**

### Step 3: Wait for Deployment

- Build takes 2-5 minutes
- You'll see progress in the dashboard
- Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

## 📁 Required Files (Already in Place)

✅ `rag_streamlit.py` - Main Streamlit app  
✅ `requirements.txt` - All dependencies  
✅ `.streamlit/config.toml` - Streamlit configuration  

## ⚙️ Required: Gemini API Key

You need to add your Gemini API key in Streamlit Cloud → Secrets:

```
GOOGLE_API_KEY = "your-gemini-api-key"
```

**Get your free API key**: https://aistudio.google.com/apikey

**Note**: 
- The app uses SentenceTransformers (free, local) for document embeddings
- Gemini API is required for document answering

## 🔄 Auto-Updates

Whenever you push to GitHub, Streamlit Cloud automatically redeploys your app!

```bash
git add .
git commit -m "Update"
git push
```

## 📚 Features

- ✅ PDF document upload
- ✅ Google Gemini API for document answering
- ✅ SentenceTransformers for embeddings (free, local)
- ✅ Source attribution
- ✅ Caching for faster processing

## 🎯 Quick Access

**Start deployment**: https://share.streamlit.io

Your app will be live in minutes! 🚀
