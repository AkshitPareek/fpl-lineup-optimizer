# Deploy to Streamlit Cloud

## Quick Start

### 1. Prepare Repository

Ensure all files are committed:
```bash
git add -A
git commit -m "Ready for Streamlit Cloud deployment"
git push origin ml-backend
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `AkshitPareek/fpl-lineup-optimizer`
5. Select branch: `ml-backend`
6. **Main file path: `streamlit_app.py`** (at root)
7. Click "Deploy"

### 3. Configuration

The app will automatically use:
- `requirements.txt` for Python packages
- `.streamlit/config.toml` for theme settings
- Port 8501 (default Streamlit port)

---

## ⚠️ Important: FPL API Limitations

**The FPL API may block requests from Streamlit Cloud due to CORS restrictions.**

### Workaround Options:

#### Option 1: Use Demo Mode (Recommended for Cloud)
- Check "Use Demo Team" checkbox
- See dashboard features with sample data
- All other features work normally

#### Option 2: Run Locally (For Full FPL API Access)
```bash
git clone https://github.com/AkshitPareek/fpl-lineup-optimizer.git
cd fpl-lineup-optimizer
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

---

## Features Available

### ✅ Works on Streamlit Cloud
- **Overview Page:** Champion model stats (EXP-032)
- **Model Lab:** Compare models, feature importance
- **Research:** Agent status, experiment results
- **Analytics:** Dataset statistics, charts
- **Team Builder (Demo):** Sample team display

### ⚠️ Requires Local Deployment
- **Team Builder (Live):** Real FPL team loading
- **Live Predictions:** Real-time player data

---

## Troubleshooting

**Issue: "Failed to load team"**
- ✅ Enable "Use Demo Team" checkbox
- FPL API blocks cross-origin requests from cloud

**Issue: "Module not found"**
- Check `requirements.txt` has all dependencies
- Push changes and redeploy

**Issue: "Model not loading"**
- `models/exp032_fdr/model.pkl` is 100MB
- May need Git LFS for large files

---

## Local Development

For full functionality including live FPL team loading:

```bash
# Clone repo
git clone https://github.com/AkshitPareek/fpl-lineup-optimizer.git
cd fpl-lineup-optimizer

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

---

## File Structure

```
fpl-lineup-optimizer/
├── streamlit_app.py          # Main entry point
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Theme settings
├── pages/                    # Page modules
│   └── 3_Team_Builder.py    # Team builder
├── models/
│   └── exp032_fdr/
│       └── model.pkl        # Champion model
└── ...
```

---

## Support

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **FPL API:** [fantasy.premierleague.com/api](https://fantasy.premierleague.com/api)
- **GitHub Issues:** [Report bugs here](https://github.com/AkshitPareek/fpl-lineup-optimizer/issues)
