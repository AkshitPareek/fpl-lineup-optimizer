# FPL Dashboard - Deployment Guide

## 🎯 Quick Start (Recommended)

### Local Deployment (Full Features)

```bash
# 1. Clone repository
git clone https://github.com/AkshitPareek/fpl-lineup-optimizer.git
cd fpl-lineup-optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run dashboard
streamlit run streamlit_app.py

# 4. Open browser
# http://localhost:8501
```

✅ **All features work locally including FPL API team loading!**

---

## ☁️ Streamlit Cloud Deployment (Limited)

**Note:** FPL API has CORS restrictions that prevent team loading from cloud.

### Deploy Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Repository: `AkshitPareek/fpl-lineup-optimizer`
5. Branch: `ml-backend`
6. Main file: `streamlit_app.py`
7. Click **"Deploy"**

### Cloud Limitations

| Feature | Local | Cloud |
|---------|-------|-------|
| Model showcase | ✅ | ✅ |
| Feature importance | ✅ | ✅ |
| Team Builder (Demo) | ✅ | ✅ |
| **Team Builder (Live FPL)** | ✅ | ❌ CORS blocked |
| Research tracking | ✅ | ✅ |
| Analytics | ✅ | ✅ |

---

## 🏆 Current Champion: EXP-032

```
Spearman: 0.7666
RMSE: 1.4389
Features: 17 (includes FDR)
Data: 75,317 samples
```

---

## 📁 File Structure

```
fpl-lineup-optimizer/
├── streamlit_app.py          # Main entry point
├── requirements.txt          # Dependencies
├── .streamlit/config.toml   # Theme settings
├── pages/                    # Additional pages
│   └── 3_Team_Builder.py    # FPL API integration
├── models/exp032_fdr/       # Champion model
└── ...
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
streamlit run streamlit_app.py --server.port 8502
```

### "FPL API timeout"
- Check internet connection
- FPL API may be down temporarily
- Try again in a few minutes

---

## 🎓 Usage

1. **Overview Page:** See EXP-032 champion stats
2. **Model Lab:** Compare models, play with predictions
3. **Team Builder:** 
   - Enter FPL Team ID
   - Click "Load Team"
   - See predicted points
   - Get transfer recommendations
4. **Research:** View experiment results
5. **Analytics:** Explore dataset

---

**Ready to use! Run locally for best experience.** 🚀
