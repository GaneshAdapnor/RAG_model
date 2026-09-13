# 🚀 Deploy to Streamlit Cloud - Step by Step

## ✅ Your Project is Ready!

**Repository**: https://github.com/GaneshAdapnor/RAG_model.git
**Main File**: `rag_streamlit.py`
**Branch**: `main`

## 📋 Deployment Steps

### Step 1: Go to Streamlit Cloud
👉 https://share.streamlit.io

### Step 2: Sign In
- Click **"Sign in"**
- Use your GitHub account

### Step 3: Create New App
- Click **"New app"** button (top right)

### Step 4: Fill in the Form

**Repository**:
- Select: `GaneshAdapnor/RAG_model`

**Branch**:
- Select: `main`

**Main file path**:
- Enter: `rag_streamlit.py`

**App URL**:
- Choose a unique name (e.g., `rag-document-qa`)
- This will be your app URL: `https://YOUR-APP-NAME.streamlit.app`

### Step 5: Add your OpenRouter API Key (CRITICAL!)

**Before clicking "Deploy":**

1. Get a free key at [OpenRouter Console](https://openrouter.ai/settings/keys). Create a few if you want automatic failover on rate limits.
2. Click **"Advanced settings"** (below the form) → **"Secrets"** tab
3. Add:
   ```
   OPENROUTER_API_KEYS = "key-one,key-two,key-three"
   ```
   (a single key also works: `OPENROUTER_API_KEY = "your-key"`)
4. Click **"Save"**

### Step 6: Deploy!
- Click **"Deploy"** button
- Wait 2-5 minutes for the build

### Step 7: Your App is Live! 🎉

Visit: `https://YOUR-APP-NAME.streamlit.app`

## ✅ Verification Checklist

Before deploying:
- ✅ Repository: `GaneshAdapnor/RAG_model`
- ✅ Branch: `main`
- ✅ Main file: `rag_streamlit.py`
- ✅ `OPENROUTER_API_KEY` / `OPENROUTER_API_KEYS` added in Secrets

## 🎯 After Deployment

1. **Test your app** by uploading a document (PDF, Word, TXT, Markdown, HTML, or CSV)
2. **Ask a question** about it in the Chat tab
3. **Confirm sources** show up alongside the answer

## 🔄 Auto-Updates

Whenever you push to GitHub, Streamlit Cloud automatically redeploys your app!

## 📚 What's Deployed

- ✅ Streamlit app powered by OpenRouter
- ✅ Multi-format document processing (PDF, Word, TXT, Markdown, HTML, CSV, RTF)
- ✅ SentenceTransformers for embeddings (free, local)
- ✅ Per-visitor session isolation (your documents aren't visible to other users)
- ✅ Automatic key failover if an OpenRouter key hits a rate limit
- ✅ Source attribution and chat history

**Your app is ready to deploy! Follow the steps above.** 🚀
