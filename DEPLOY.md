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
6. Main file path: `dashboard/app.py`
7. Click "Deploy"

### 3. Configuration

The app will automatically use:
- `requirements.txt` for Python packages
- `.streamlit/config.toml` for theme settings
- Port 8501 (default Streamlit port)

### 4. Environment Variables (if needed)

If you need API keys or secrets:
1. Go to app settings on Streamlit Cloud
2. Click "Secrets"
3. Add in TOML format:
```toml
[fpl]
api_key = "your_key_here"
```

### 5. Features Enabled

✅ **FPL API Integration**
- Load real team data by Team ID
- Live player statistics
- Transfer recommendations
- Predicted points using EXP-032 model

✅ **Model Showcase**
- EXP-032 champion model (Spearman 0.7666)
- Feature importance visualization
- Model comparison (EXP-030 vs EXP-031 vs EXP-032)

✅ **Interactive Team Builder**
- Enter FPL Team ID
- See predicted points for your squad
- Get transfer suggestions
- View top targets

✅ **Research Tracking**
- Agent status
- Experiment results
- Ralph Loop progress

### 6. Troubleshooting

**Issue: "Module not found"**
- Check `requirements.txt` includes all dependencies
- Push changes and redeploy

**Issue: "Model not loading"**
- Ensure `models/exp032_fdr/model.pkl` is in repo
- File is 100MB, may need Git LFS

**Issue: "FPL API timeout"**
- API calls are cached for 5 minutes
- Normal during high traffic

### 7. Custom Domain (Optional)

To use custom domain:
1. Go to app settings
2. Click "Custom domain"
3. Follow DNS configuration steps

## Local Development

```bash
cd dashboard
streamlit run app.py
```

Access at: http://localhost:8501

## Support

- Streamlit Docs: [docs.streamlit.io](https://docs.streamlit.io)
- FPL API: [fantasy.premierleague.com/api](https://fantasy.premierleague.com/api)
