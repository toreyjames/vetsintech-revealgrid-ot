# RevealGrid Investor Demo

A turn-key, investor-facing Streamlit mini-app demonstrating RevealGrid's AI-driven industrial intelligence platform for OT environments.

## 🚀 Quick Start

```bash
git clone https://github.com/your-org/revealgrid_demo
cd revealgrid_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload the provided sample `sample_data/cmdb_sample.json`, click **Run discovery**, and explore the tabs.

## 📋 What Investors Will Experience

1. **ChatOps Interface** – familiar conversational experience like ChatGPT
2. **Progressive Discovery** – "What's your plant name?" → "Upload CMDB or skip" → "Upload P&ID or skip"
3. **Instant Visual Payoff** – global map appears within 10 seconds of first site entry
4. **Natural Commands** – type "insight" for recommendations, "add" for another site, "map" for visualization
5. **Graceful Skips** – every step can be skipped to keep the demo moving

## 🏗️ Architecture

```
revealgrid_demo/
├── app.py                # Streamlit UI
├── revealgrid_core.py    # Domain logic (graph, AI stubs, exports)
├── sample_data/
│   ├── pid_sample.pdf
│   └── cmdb_sample.json
├── requirements.txt
└── README.md
```

## 🎯 Demo Flow

### 1️⃣ ChatOps Discovery
- **"What's your plant name and location?"** → Automatic geocoding
- **"Upload CMDB or type skip"** → Asset synchronization
- **"Upload P&ID or type skip"** → Document parsing
- **"Mapping complete!"** → Instant visual payoff

### 2️⃣ Natural Commands
- **"insight"** → Downtime & risk recommendations
- **"add"** → Map another site
- **"map"** → View global site visualization
- **"process"** → See plant flow diagram

### 3️⃣ Export Model
- Download encrypted world model stub
- Demonstrates secure data export capability

## 🔧 Technical Features

- **Clean first-run experience** – no raw Base-64; visuals render instantly
- **Wizard-style onboarding** – investors can walk through "Connect → Upload → Discover → Explore" in < 5 minutes
- **Modular codebase** – separates lightweight discovery from heavier AI so you can refactor later without breaking the UI
- **Simple deployment** – runs locally with `streamlit run app.py` or on Streamlit Community Cloud

## 🚀 Next Hardening Steps (Post-Demo)

| Area | Action |
|------|--------|
| **Branding** | Add company logo in `st.sidebar.image()` |
| **Real encryption** | Swap the Base-64 stub for `cryptography.fernet.Fernet` payload |
| **Modular packaging** | Publish `revealgrid_core` as a separate pip package for future micro-service split |
| **CI / demo URL** | Push to *Streamlit Community Cloud* (`streamlit.io/cloud`) so investors can play without cloning code |
| **Live data hook** | After PoC, replace `sample_graph()` with a thin client hitting your edge gateway's REST API |

## 📦 Dependencies

- `streamlit>=1.35` - Web app framework
- `networkx` - Graph operations
- `matplotlib` - Visualizations
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `torch` - CV stub (comment out if CPU-only demo)
- `cryptography` - Future AES/Fernet export

## 🎨 Customization

### Adding Company Branding
```python
# In app.py, add after st.title():
st.sidebar.image("path/to/logo.png", width=200)
```

### Custom Sample Data
Replace `sample_data/cmdb_sample.json` with your actual ServiceNow export format.

### Enhanced Visualizations
Modify `revealgrid_core.py` functions to add more sophisticated analytics.

## 🔒 Security Notes

- Current export uses Base-64 encoding (demo stub)
- Production should use proper encryption with `cryptography.fernet.Fernet`
- Consider adding authentication for live deployments

## 📞 Support

For questions about the demo or RevealGrid platform, contact the development team. 