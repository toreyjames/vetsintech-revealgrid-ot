# RevealGrid Demo - VetinTech Event Deployment Guide

## 🚀 Quick Deploy Options

### Option 1: Streamlit Cloud (Recommended)
1. **Fork this repository** to your GitHub account
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Deploy the `simple_demo.py` file**
5. **Get your public URL** (e.g., `https://revealgrid-demo.streamlit.app`)

### Option 2: Heroku
```bash
# Create Procfile
echo "web: streamlit run simple_demo.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create requirements.txt
echo "streamlit==1.24.0
networkx==2.8.8
matplotlib==3.5.3
numpy==1.21.6" > requirements.txt

# Deploy to Heroku
heroku create revealgrid-demo
git add .
git commit -m "Deploy RevealGrid demo"
git push heroku main
```

### Option 3: Railway
1. **Go to [railway.app](https://railway.app)**
2. **Connect GitHub repository**
3. **Select `simple_demo.py` as the entry point**
4. **Deploy automatically**

### Option 4: Local Demo
```bash
# In the revealgrid_demo directory
streamlit run simple_demo.py --server.port 8501
```

## 📋 Demo Features

### 🎯 **Step-by-Step Walkthrough**
1. **Welcome & Overview** - Platform introduction
2. **Asset Discovery** - Simulated industrial asset detection
3. **Risk Analysis** - Vulnerability assessment with severity levels
4. **Automation Recommendations** - AI-powered optimization suggestions
5. **World Model Export** - Secure model generation and download

### 🔧 **Key Capabilities Demonstrated**
- ✅ **Hybrid Asset Discovery** (Network + Documents + Computer Vision)
- ✅ **Risk Assessment** with severity classification
- ✅ **AI-Powered Recommendations** with reward scoring
- ✅ **World Model Simulation** with performance optimization
- ✅ **Secure Export** with encrypted data
- ✅ **Air-Gapped Compliance** ready for OT environments

### 📊 **Interactive Elements**
- **Real-time asset discovery simulation**
- **Dynamic risk analysis with charts**
- **AI recommendation quality metrics**
- **Performance optimization visualization**
- **Encrypted model export and download**

## 🎪 Event Presentation Tips

### **Opening (30 seconds)**
> *"RevealGrid is an AI-driven industrial intelligence platform that automatically discovers, analyzes, and optimizes industrial assets in OT environments. Today I'll show you how it works."*

### **Demo Flow (3-4 minutes)**
1. **Click "Start Demo"** - Show the overview
2. **Click "Start Discovery"** - Watch assets being discovered
3. **Click "Analyze Risks"** - Show risk assessment
4. **Click "Generate Recommendations"** - Show AI insights
5. **Click "Export Encrypted Model"** - Show secure export

### **Key Talking Points**
- **"Hybrid discovery combines network scans, document parsing, and computer vision"**
- **"Risk analysis identifies vulnerabilities with severity levels"**
- **"AI recommendations come with confidence scores"**
- **"World model simulation optimizes performance"**
- **"Encrypted exports ensure security compliance"**

### **Closing (30 seconds)**
> *"RevealGrid is ready for production deployment in air-gapped OT environments, providing the intelligence needed for modern industrial operations."*

## 🔗 **Public Demo URLs**

Once deployed, you can share these URLs:

### **Streamlit Cloud**
- `https://your-username-revealgrid-demo.streamlit.app`

### **Heroku**
- `https://revealgrid-demo.herokuapp.com`

### **Railway**
- `https://revealgrid-demo.railway.app`

## 📱 **Mobile-Friendly**
The demo is fully responsive and works on:
- ✅ Desktop browsers
- ✅ Tablets
- ✅ Mobile phones
- ✅ Touch interfaces

## 🎨 **Customization Options**

### **Change Colors**
Edit the CSS in `simple_demo.py`:
```css
.main-header {
    border-bottom: 3px solid #YOUR_COLOR;
}
```

### **Add Your Logo**
Replace the header with your logo:
```python
st.image("your_logo.png", width=200)
```

### **Custom Branding**
Update the footer text and company information.

## 🚨 **Troubleshooting**

### **If demo doesn't load:**
1. Check browser console for errors
2. Try refreshing the page
3. Clear browser cache
4. Try different browser

### **If deployment fails:**
1. Check requirements.txt has all dependencies
2. Ensure Python version compatibility
3. Verify file paths are correct
4. Check platform-specific deployment guides

## 📞 **Support**

For deployment issues:
- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **Railway**: [docs.railway.app](https://docs.railway.app)

## 🎯 **VetinTech Event Checklist**

- [ ] Deploy demo to public URL
- [ ] Test on different devices
- [ ] Practice demo flow
- [ ] Prepare talking points
- [ ] Have backup plan (local demo)
- [ ] Share URL with event organizers
- [ ] Test internet connection at venue

---

**Good luck with your VetinTech presentation! 🚀** 