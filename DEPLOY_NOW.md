# 🚀 Deploy to Streamlit Cloud - Step by Step

## ✅ Your Project is Ready!

**Repository**: https://github.com/GaneshAdapnor/RAG_model.git  
**Main File**: `rag_streamlit.py`  
**Branch**: `main`

## 📋 Deployment Steps

### Step 1: Go to Streamlit Cloud
👉 **Already opened in your browser**: https://share.streamlit.io

### Step 2: Sign In
- Click **"Sign in"**
- Use your GitHub account: **GaneshAdapnor**

### Step 3: Create New App
- Click **"New app"** button (top right)

### Step 4: Fill in the Form

**Repository**:
- Search for: `RAG_model`
- Or select: `GaneshAdapnor/RAG_model`

**Branch**:
- Select: `main`

**Main file path**:
- Enter: `rag_streamlit.py`

**App URL**:
- Choose a unique name (e.g., `rag-pdf-qa` or `rag-document-qa`)
- This will be your app URL: `https://YOUR-APP-NAME.streamlit.app`

### Step 5: Add API Key (CRITICAL!)

**Before clicking "Deploy":**

1. Click **"Advanced settings"** (below the form)
2. Click **"Secrets"** tab
3. Click **"New secret"**
4. Enter:
   - **Key**: `GOOGLE_API_KEY`
   - **Value**: `AIzaSyD_wUrMD5bY6Rj4iDroVeCpBeM9RFxICbA`
5. Click **"Add"**
6. Click **"Save"**

### Step 6: Deploy!
- Click **"Deploy"** button
- Wait 2-5 minutes for the build

### Step 7: Your App is Live! 🎉

Once deployed, you'll see: **"Your app is live!"**
- Visit: `https://YOUR-APP-NAME.streamlit.app`

## ✅ Verification Checklist

Before deploying:
- ✅ Repository: `GaneshAdapnor/RAG_model`
- ✅ Branch: `main`
- ✅ Main file: `rag_streamlit.py`
- ✅ API Key in Secrets: `GOOGLE_API_KEY = AIzaSyD_wUrMD5bY6Rj4iDroVeCpBeM9RFxICbA`

## 📝 Your API Key

```
GOOGLE_API_KEY = AIzaSyD_wUrMD5bY6Rj4iDroVeCpBeM9RFxICbA
```

## 🎯 After Deployment

1. **Test your app** by uploading a PDF
2. **Select model**: `gemini-1.5-flash` (default)
3. **Ask a question** about the PDF
4. **Verify it works** with Gemini API

## 🔄 Auto-Updates

Whenever you push to GitHub, Streamlit Cloud automatically redeploys your app!

## 📚 What's Deployed

- ✅ Streamlit app with Gemini API
- ✅ PDF document processing
- ✅ SentenceTransformers for embeddings (free)
- ✅ Source attribution
- ✅ Clean project structure

**Your app is ready to deploy! Follow the steps above.** 🚀
