# 🚀 Deploy to Streamlit Cloud - Step by Step Guide

## Your app is ready! Follow these steps:

### Step 1: Push to GitHub

Run these commands in your terminal:

```bash
cd C:\Users\gadap\OneDrive\Projects\LLM
git add rag_streamlit.py requirements.txt .streamlit/config.toml .gitignore DEPLOYMENT.md STREAMLIT_DEPLOY.md packages.txt
git commit -m "RAG PDF Q&A System - Ready for Streamlit Cloud"
git push origin main
```

**Note**: If you want a separate repository for this app:
1. Create a new repository on GitHub: https://github.com/new
2. Name it: `rag-pdf-qa` (or any name you like)
3. Then run:
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/rag-pdf-qa.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"** button
4. **Fill in the form**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `rag_streamlit.py`
   - **App URL**: Choose a unique name (e.g., `rag-pdf-qa` or `your-name-rag`)
5. **Click "Advanced settings"** → **"Secrets"** (optional):
   - Add your OpenAI API key if you want to use OpenAI embeddings:
     ```
     OPENAI_API_KEY = "your-api-key-here"
     ```
   - **Note**: If you don't add this, the app will use SentenceTransformers (free, local)
6. **Click "Deploy"**

### Step 3: Wait for Deployment

- Build time: 2-5 minutes
- You'll see progress in the dashboard
- When done, you'll see "Your app is live!"

### Step 4: Access Your App

Your app will be available at:
```
https://YOUR-APP-NAME.streamlit.app
```

## ✅ What's Already Prepared

- ✅ `rag_streamlit.py` - Main app file
- ✅ `requirements.txt` - All dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ Error handling for invalid API keys
- ✅ Fallback to SentenceTransformers if no API key

## 💡 Tips

1. **Free to use**: Streamlit Cloud is free
2. **No API key needed**: App works with SentenceTransformers (free, local)
3. **Auto-update**: Pushing to GitHub automatically redeploys
4. **Share your app**: Share the URL with anyone!

## 📝 Need Help?

- Check `DEPLOYMENT.md` for detailed instructions
- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-cloud
- Check build logs in Streamlit Cloud dashboard if deployment fails

## 🎉 You're Ready!

Just push to GitHub and deploy on Streamlit Cloud. Your app will be live in minutes!

