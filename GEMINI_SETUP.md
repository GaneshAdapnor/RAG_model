# 🚀 How to Use Gemini API Instead of OpenAI

## ✅ Gemini API Support Added!

I've added **Google Gemini API** support to your RAG app. You can now use Gemini instead of OpenAI!

## 🎯 Benefits of Using Gemini

1. **Free Tier Available** - Gemini offers generous free tier
2. **No Quota Issues** - Better than hitting OpenAI quota limits
3. **High Quality** - Gemini models are very capable
4. **Easy to Get** - Free API key from Google AI Studio

## 📋 How to Use Gemini

### Step 1: Get Your Gemini API Key (Free!)

1. **Go to**: https://aistudio.google.com/apikey
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy your API key**

### Step 2: Use Gemini in the App

1. **Open your Streamlit app**
2. **In the sidebar**, select **"Google Gemini"** from **"LLM Provider"** dropdown
3. **Enter your Gemini API key** in the **"Google Gemini API Key"** field
4. **Choose a Gemini model**:
   - `gemini-1.5-flash` (fast, recommended)
   - `gemini-1.5-pro` (more capable)
   - `gemini-pro` (standard)
   - `gemini-2.0-flash-exp` (experimental)
5. **Upload your PDFs** and ask questions!

### Step 3: Embeddings (Optional)

You can still use:
- **OpenAI embeddings** (if you have OpenAI API key)
- **SentenceTransformers** (free, local) - default

## 🔄 Switching Between Providers

You can easily switch between OpenAI and Gemini:
- **Select provider** from dropdown in sidebar
- **Enter API key** for the selected provider
- **Choose model** for that provider
- **That's it!** The app will use the selected provider

## 💡 Current Configuration

- **LLM Provider**: Choose between OpenAI or Google Gemini
- **Embeddings**: Can use OpenAI or SentenceTransformers (free)
- **Models Available**:
  - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo-preview, etc.
  - Gemini: gemini-1.5-flash, gemini-1.5-pro, gemini-pro, etc.

## 📝 What Changed

✅ Added `langchain-google-genai` to requirements.txt
✅ Added Gemini import and availability check
✅ Added provider selection dropdown
✅ Added Gemini API key input
✅ Added Gemini model selection
✅ Updated LLM initialization to support both providers
✅ Updated error handling for both providers

## 🚀 Deployment Status

✅ **All changes pushed to GitHub**
✅ **Streamlit Cloud will auto-redeploy** (1-2 minutes)
✅ **Gemini support will be available** after deployment

## 🎉 You're All Set!

1. **Wait for auto-redeploy** (1-2 minutes)
2. **Get your Gemini API key**: https://aistudio.google.com/apikey
3. **Use Gemini** in the app - no more quota issues!

## Resources

- **Get Gemini API Key**: https://aistudio.google.com/apikey
- **Gemini Documentation**: https://ai.google.dev/gemini-api/docs
- **LangChain Gemini**: https://python.langchain.com/docs/integrations/chat/google_generative_ai

