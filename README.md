# RevealGrid OT - Asset Discovery & Process Understanding Platform

## Project Overview

RevealGrid OT is an innovative asset discovery and process understanding platform designed for industrial environments, with a focus on Toyota-style automotive manufacturing simulation. This platform provides real-time visualization of manufacturing assets, process flows, and automation recommendations.

## Key Features

### Interactive Global Site Map
- Real-time visualization of manufacturing plants across the USA
- Color-coded efficiency indicators
- Interactive plant selection and drill-down capabilities

### Asset Discovery & Inventory
- Comprehensive asset tracking (PLC Controllers, HMI Systems, Robots, Sensors, etc.)
- Real-time asset health monitoring
- Performance metrics and trend analysis

### Process Flow Visualization
- Interactive process flow diagrams
- Manufacturing step visualization
- Bottleneck identification and optimization recommendations

### Automation Intelligence
- AI-powered automation recommendations
- Predictive maintenance suggestions
- Performance optimization insights

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Visualization**: Matplotlib, NetworkX, Plotly
- **Mapping**: Streamlit Maps, PyDeck
- **Data Processing**: Pandas, NumPy
- **Process Modeling**: NetworkX (graph theory)

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RevealGrid-OT
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically load with the interactive map

## Usage Guide

### Navigating the Application

1. **Global Map View**
   - View all manufacturing plants on the interactive map
   - Each plant is color-coded based on efficiency (Green: High, Yellow: Medium, Red: Low)

2. **Plant Selection**
   - Use the radio buttons to select a specific plant
   - Click on any plant to view detailed analysis

3. **Asset Analysis**
   - View comprehensive asset inventory
   - Analyze asset distribution with pie charts
   - Monitor asset health status

4. **Process Flow**
   - Visualize manufacturing processes
   - Identify bottlenecks and optimization opportunities
   - Track process efficiency metrics

5. **Automation Recommendations**
   - Receive AI-powered suggestions for improvements
   - View predictive maintenance recommendations
   - Analyze automation opportunities

## Demo Data

The application includes simulated data for four Toyota manufacturing plants:

- **Plant A - Detroit**: High automation, good efficiency
- **Plant B - Chicago**: Highest automation level, best performance
- **Plant C - Austin**: Lower automation, needs improvements
- **Kentucky Plant**: Balanced performance, comprehensive processes

## Deployment Options

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment (Heroku)
1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy to Heroku:
   ```bash
   heroku create revealgrid-ot
   git push heroku main
   ```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Project Structure

```
RevealGrid-OT/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Sample data and datasets
├── models/               # ML models and algorithms
└── notebooks/            # Jupyter notebooks for development
```

## Future Enhancements

- Real-time IoT data integration
- Advanced machine learning algorithms
- Mobile application development
- Multi-tenant architecture
- Advanced security features

## Contributing

This project was developed for VetsInTech submission. For questions or contributions, please contact the development team.

## License

This project is proprietary software developed for VetsInTech competition.

## Contact

For technical support or questions about this project, please reach out to the development team.

---

**Developed for VetsInTech 2024** 