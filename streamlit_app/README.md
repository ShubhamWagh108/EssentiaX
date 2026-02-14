# 🧠 EssentiaX Streamlit Dashboard

Advanced Exploratory Data Analysis (EDA) Dashboard for Data Scientists and Analysts

## 🌟 Features

### 🚀 **Smart Data Handling**
- **File Upload**: Support for CSV, Excel files up to 200MB
- **URL/Path Loading**: Load data from URLs or local file paths
- **Sample Datasets**: Built-in sample datasets for testing
- **Progressive Loading**: Efficient loading of large datasets (1GB+)
- **Smart Sampling**: Preserves data distribution while reducing size

### 📊 **Advanced EDA Capabilities**
- **15+ Statistical Tests**: Normality, correlation, hypothesis testing
- **AI-Powered Insights**: Plain English interpretations of statistical results
- **Data Quality Scoring**: 5-dimensional quality assessment
- **Problem Type Detection**: Automatic classification/regression/NLP detection
- **Outlier Detection**: Multi-method consensus outlier identification

### 🎨 **Interactive Visualizations**
- **20+ Plot Types**: Distribution, correlation, diagnostic plots
- **Big Data Optimized**: Handles large datasets through smart sampling
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Real-time Updates**: Dynamic plot generation based on user selections

### 🤖 **AI Recommendations**
- **Preprocessing Guidance**: Specific steps for data cleaning
- **Feature Engineering**: Automated feature engineering suggestions
- **Model Selection**: ML model recommendations based on data characteristics
- **Business Insights**: Translation of statistical findings to business value

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or download the EssentiaX project**
   ```bash
   git clone <repository-url>
   cd EssentiaX
   ```

2. **Install dependencies**
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   python run_app.py
   ```
   
   Or manually:
   ```bash
   streamlit run main.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL

## 🚀 Quick Start Guide

### 1. **Upload Your Data**
- Navigate to "🏠 Home & Data Upload"
- Choose from three upload methods:
  - **📎 File Upload**: Drag & drop or browse for files
  - **🔗 URL/Path**: Enter file URL or local path
  - **🎲 Sample Dataset**: Use built-in sample data

### 2. **Run EDA Analysis**
- Go to "📊 Quick EDA Overview"
- Configure analysis settings:
  - Select target column (optional)
  - Set sample size for large datasets
  - Enable/disable advanced features
- Click "🚀 Run Complete EDA Analysis"

### 3. **Explore Results**
- **📋 Summary**: Dataset overview and key metrics
- **📊 Statistics**: Detailed statistical analysis
- **🤖 AI Insights**: Recommendations and interpretations

### 4. **Advanced Analysis** (Coming Soon)
- **🔬 Advanced Statistical Analysis**: Deep statistical testing
- **📈 Interactive Visualizations**: Custom plot generation
- **📋 Export & Reports**: Generate comprehensive reports

## 📊 Supported Data Formats

| Format | Extension | Max Size | Notes |
|--------|-----------|----------|-------|
| CSV | `.csv` | 200MB | Auto-delimiter detection |
| Excel | `.xlsx`, `.xls` | 200MB | Multi-sheet support |
| URLs | HTTP/HTTPS | 200MB | Direct URL loading |

## 🎯 Use Cases

### **Data Scientists**
- Quick dataset exploration and profiling
- Statistical significance testing
- Feature engineering guidance
- Model selection recommendations

### **Business Analysts**
- Data quality assessment
- Business insight generation
- Automated report creation
- Non-technical interpretations

### **Students & Researchers**
- Learning statistical concepts
- Exploring sample datasets
- Understanding data distributions
- Hypothesis testing practice

## 🔧 Configuration

### **Memory Settings**
- Default max memory: 500MB
- Configurable in `utils/data_handler.py`
- Smart sampling activates for large datasets

### **Streamlit Settings**
- Configuration file: `.streamlit/config.toml`
- Default port: 8501
- Max upload size: 200MB

### **Advanced Features**
- Enable/disable advanced statistics
- Toggle AI insights
- Customize sample sizes
- Configure plot limits

## 📈 Performance Optimization

### **Large Dataset Handling**
- **Smart Sampling**: Maintains statistical properties
- **Progressive Loading**: Load data in chunks
- **Memory Monitoring**: Real-time memory usage tracking
- **Chunked Processing**: Process data in manageable pieces

### **Visualization Optimization**
- **Plot Sampling**: Reduce points for better performance
- **Interactive Caching**: Cache plot data for faster updates
- **Lazy Loading**: Load visualizations on demand

## 🐛 Troubleshooting

### **Common Issues**

**1. Import Errors**
```
Error: EssentiaX modules not available
```
**Solution**: Ensure you're running from the correct directory and EssentiaX is installed

**2. Memory Issues**
```
Warning: Memory usage exceeds limit
```
**Solution**: Reduce sample size or enable smart sampling

**3. File Upload Errors**
```
Error: File too large
```
**Solution**: Use progressive loading or reduce file size

**4. Visualization Issues**
```
Error: Cannot create plots
```
**Solution**: Check if Plotly is installed and data is loaded

### **Performance Tips**
- Use sample datasets for testing
- Enable smart sampling for large files
- Close unused browser tabs
- Monitor memory usage in sidebar

## 🔄 Updates and Maintenance

### **Version Information**
- Current version: 1.0.0
- Last updated: January 2024
- Compatibility: Python 3.8+

### **Update Process**
1. Pull latest changes from repository
2. Update dependencies: `pip install -r requirements.txt --upgrade`
3. Restart the application

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests: `python test_streamlit_app.py`
5. Submit pull request

### **Code Structure**
```
streamlit_app/
├── main.py                 # Main Streamlit application
├── utils/
│   └── data_handler.py     # Data processing utilities
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── requirements.txt        # Python dependencies
├── run_app.py             # App launcher script
└── README.md              # This file
```

## 📞 Support

### **Getting Help**
- Check the troubleshooting section above
- Review the EssentiaX documentation
- Submit issues on the project repository

### **Feature Requests**
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Provide examples if possible

## 📄 License

This project is part of the EssentiaX library. Please refer to the main project license for terms and conditions.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **Pandas & NumPy**: For data processing capabilities
- **SciPy & Scikit-learn**: For statistical analysis

---

**🚀 Ready to explore your data? Launch the dashboard and start analyzing!**

```bash
cd streamlit_app
python run_app.py
```