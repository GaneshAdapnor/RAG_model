# 🔍 White Screen - How to Debug

## What a White Screen Means

A **white screen** on Streamlit Cloud usually means:
1. ⏳ **App is still loading** - Wait 30-60 seconds
2. ❌ **Import error** - Check build logs
3. 💥 **Runtime error** - Check application logs  
4. 🔄 **App crashed during startup**

## ✅ How to Check What's Wrong

### Step 1: Check Streamlit Cloud Logs

1. Go to: https://share.streamlit.io
2. Click on your app
3. Click **"Manage app"** (gear icon)
4. Click **"Logs"** tab
5. Look for:
   - ❌ Red error messages
   - ⚠️ Warning messages
   - 📝 Build errors

### Step 2: Check Browser Console

1. Press **F12** (or right-click → Inspect)
2. Click **"Console"** tab
3. Look for:
   - Red error messages
   - Network errors
   - JavaScript errors

### Step 3: Check Build Status

1. In Streamlit Cloud dashboard
2. Check if status shows:
   - ✅ **Running** - App is live
   - ⏳ **Building** - Still deploying
   - ❌ **Failed** - Check logs

## 🔧 Common Fixes

### Issue 1: Import Error
**Symptom**: `ModuleNotFoundError` in logs
**Fix**: All dependencies are in `requirements.txt` ✅

### Issue 2: Still Building
**Symptom**: White screen, no errors
**Fix**: Wait 2-5 minutes for first deployment

### Issue 3: Memory Error
**Symptom**: Out of memory in logs
**Fix**: Use OpenAI API key instead of SentenceTransformers

### Issue 4: Page Config Error
**Symptom**: `set_page_config` error
**Fix**: Already fixed in code ✅

## 🚀 Quick Fixes Applied

I've added:
- ✅ Better error handling
- ✅ Error messages in app (if it crashes)
- ✅ Import error handling
- ✅ Cache directory error handling

## 📋 Next Steps

1. **Check the logs** in Streamlit Cloud dashboard
2. **Share the error message** if you see one
3. **Wait 1-2 minutes** if it's still building
4. **Hard refresh** browser (Ctrl+F5)

## 💡 If Still White Screen

1. Go to Streamlit Cloud → Your App → Manage → Logs
2. Copy the error message
3. Share it so we can fix it

The updated code should now show error messages instead of a white screen!

