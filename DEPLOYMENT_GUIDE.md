# RevealGrid OT - Deployment Guide for VetsInTech

## Quick Deployment Options

### Option 1: Streamlit Cloud (Recommended for VetsInTech)

1. **Prepare your repository:**
   ```bash
   git add .
   git commit -m "Ready for VetsInTech submission"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path to: `app.py`
   - Click "Deploy"

3. **Your app will be live at:** `https://your-app-name.streamlit.app`

### Option 2: Heroku Deployment

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku:**
   ```bash
   heroku login
   ```

3. **Create Heroku app:**
   ```bash
   heroku create revealgrid-ot-vetsintech
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

5. **Open your app:**
   ```bash
   heroku open
   ```

### Option 3: Railway Deployment

1. **Go to [railway.app](https://railway.app)**
2. **Connect your GitHub repository**
3. **Set environment variables:**
   - `PORT`: 8501
4. **Deploy automatically**

### Option 4: Render Deployment

1. **Go to [render.com](https://render.com)**
2. **Create new Web Service**
3. **Connect your GitHub repository**
4. **Set build command:** `pip install -r requirements.txt`
5. **Set start command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Local Development Setup

### Prerequisites
- Python 3.8+
- Git

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd RevealGrid-OT
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally:**
   ```bash
   streamlit run app.py
   ```

5. **Access at:** `http://localhost:8501`

## Docker Deployment

### Build and Run with Docker

1. **Build the image:**
   ```bash
   docker build -t revealgrid-ot .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 revealgrid-ot
   ```

3. **Access at:** `http://localhost:8501`

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  revealgrid-ot:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

Run with:
```bash
docker-compose up
```

## Testing Your Deployment

### Before Submission

1. **Test all features:**
   - [ ] Interactive map loads correctly
   - [ ] Plant selection works
   - [ ] Asset details display properly
   - [ ] Process flow diagrams render
   - [ ] Recommendations show up

2. **Check mobile responsiveness:**
   - Test on different screen sizes
   - Ensure touch interactions work

3. **Performance test:**
   - Load time under 10 seconds
   - Smooth interactions
   - No errors in browser console

## VetsInTech Submission Checklist

### Required Files
- [ ] `app.py` - Main application
- [ ] `requirements.txt` - Dependencies
- [ ] `README.md` - Project documentation
- [ ] `Procfile` - Heroku deployment
- [ ] `Dockerfile` - Container deployment
- [ ] `.gitignore` - Repository cleanup

### Documentation
- [ ] Clear project description
- [ ] Installation instructions
- [ ] Usage guide
- [ ] Technology stack explanation
- [ ] Deployment options

### Demo Features
- [ ] Interactive map visualization
- [ ] Asset discovery and inventory
- [ ] Process flow analysis
- [ ] Automation recommendations
- [ ] Professional UI/UX

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Dependencies not found:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Map not loading:**
   - Check internet connection
   - Verify Streamlit version compatibility

4. **Heroku deployment fails:**
   - Ensure `Procfile` is in root directory
   - Check `requirements.txt` format
   - Verify Python version compatibility

### Performance Optimization

1. **Reduce load time:**
   - Minimize dependencies
   - Use efficient data structures
   - Cache expensive computations

2. **Memory optimization:**
   - Close matplotlib figures
   - Use generators for large datasets
   - Implement lazy loading

## Support

For deployment issues or questions:
- Check the [Streamlit documentation](https://docs.streamlit.io)
- Review [Heroku deployment guide](https://devcenter.heroku.com/articles/python-support)
- Contact the development team

---

**Ready for VetsInTech 2024 Submission!** 