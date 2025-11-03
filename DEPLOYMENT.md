# Deploy to Streamlit Cloud

This guide will help you deploy the RAG PDF Q&A System to Streamlit Cloud (free hosting).

## Prerequisites

1. **GitHub Account** - Streamlit Cloud connects to GitHub repositories
2. **Git Repository** - Your code should be in a GitHub repository
3. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)

```bash
git init
git add .
git commit -m "Initial commit: RAG PDF Q&A System"
```

### 1.2 Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it (e.g., `rag-pdf-qa`)
3. Don't initialize with README (if you already have files)

### 1.3 Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/rag-pdf-qa.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Sign in to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"

### 2.2 Configure Your App

Fill in the deployment form:

- **Repository**: Select your GitHub repository (`rag-pdf-qa`)
- **Branch**: `main` (or `master`)
- **Main file path**: `rag_streamlit.py`
- **App URL**: Choose a unique name (e.g., `your-name-rag-pdf-qa`)

### 2.3 Add Secrets (API Keys)

Click "Advanced settings" and add your secrets:

**For OpenAI API (optional):**
```
OPENAI_API_KEY = "your-api-key-here"
```

**Note**: If you don't add OPENAI_API_KEY, the app will use SentenceTransformers (free, local embeddings)

### 2.4 Deploy

Click "Deploy" and wait for the build to complete (usually 2-5 minutes).

## Step 3: Configuration

### Environment Variables (Optional)

You can set these in Streamlit Cloud secrets:

- `OPENAI_API_KEY` - Your OpenAI API key (optional)
- `HTTP_PROXY` - Proxy URL if needed
- `HTTPS_PROXY` - Proxy URL if needed

### File Structure for Deployment

Your repository should have:
```
├── rag_streamlit.py      # Main app file
├── requirements.txt      # Dependencies
├── .streamlit/
│   └── config.toml       # Streamlit config
├── .gitignore           # Git ignore file
└── README.md            # Documentation
```

## Step 4: Access Your App

Once deployed, your app will be available at:
```
https://your-app-name.streamlit.app
```

## Troubleshooting

### Build Fails

1. **Check requirements.txt** - Ensure all dependencies are listed
2. **Check Python version** - Streamlit Cloud uses Python 3.11 by default
3. **Check logs** - View build logs in Streamlit Cloud dashboard

### App Errors

1. **Missing dependencies** - Add to `requirements.txt`
2. **API key issues** - Check secrets are set correctly
3. **Memory limits** - SentenceTransformers uses ~500MB RAM

### Common Issues

**Issue**: "ModuleNotFoundError"
**Solution**: Add the missing package to `requirements.txt`

**Issue**: "AuthenticationError" for OpenAI
**Solution**: Check OPENAI_API_KEY in Streamlit Cloud secrets

**Issue**: Slow loading
**Solution**: First load downloads SentenceTransformers model (~80MB). Subsequent loads are faster.

## Updating Your App

1. Make changes to your code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```
3. Streamlit Cloud automatically redeploys on push

## Cost

- **Streamlit Cloud**: Free (with limitations)
- **OpenAI API**: Pay-per-use (if using OpenAI embeddings)
- **SentenceTransformers**: Free (runs locally)

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Secrets](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
- [GitHub Guides](https://guides.github.com/)

## Notes

- The app uses FAISS for vector storage (runs in memory)
- Uploaded files are stored temporarily (not persisted between sessions)
- Cache directory (`.rag_cache`) is not persisted on Streamlit Cloud
- For production use, consider persistent storage solutions

Happy deploying! 🚀

