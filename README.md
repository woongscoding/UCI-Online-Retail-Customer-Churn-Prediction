# 🛒 Customer Churn Prediction using RFM Analysis
**E-commerce Customer Retention Strategy through Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Project Overview

This project develops a customer churn prediction model for an online retail business using **RFM Analysis** and **Machine Learning**. By analyzing customer purchase behavior patterns, we can proactively identify customers at risk of churning and implement targeted retention strategies.

### 🎯 Key Achievements
- **86.1% ROC-AUC** performance with stable cross-validation results
- **Identified 304 high-risk customers** representing £180,953 potential revenue loss
- **Discovered and resolved data leakage** ensuring fair model evaluation
- **1,418% ROI** potential through targeted marketing campaigns

## 🗂️ Project Structure

```
customer-churn-prediction/
├── 📄 README.md                     # Project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📁 data/
│   ├── raw/                         # Original UCI dataset
│   └── processed/                   # Cleaned and processed data
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_rfm_analysis.ipynb        # RFM segmentation
│   ├── 03_feature_engineering.ipynb # Feature creation
│   └── 04_model_development.ipynb   # ML model training
├── 📁 src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning functions
│   ├── rfm_analysis.py             # RFM calculation functions
│   ├── feature_engineering.py      # Feature creation
│   └── model_utils.py              # ML utilities
├── 📁 results/
│   ├── final_rfm_analysis.csv       # Customer segments
│   ├── model_performance.csv        # Model comparison
│   └── feature_importance.csv       # Feature analysis
├── 📁 visualizations/
│   ├── rfm_segments.png            # Customer segmentation
│   ├── model_comparison.png         # Performance comparison
│   └── feature_importance.png       # Feature analysis
└── 📁 docs/
    ├── methodology.md               # Technical methodology
    └── business_impact.md           # Business case study
```

## 📈 Dataset Information

**Source**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Period**: December 2010 - December 2011
- **Records**: 354,321 transactions (after cleaning)
- **Customers**: 3,920 unique customers
- **Geography**: UK-based online retail transactions

## 🔍 Methodology

### 1. RFM Analysis
- **Recency**: Days since last purchase
- **Frequency**: Number of transactions
- **Monetary**: Total purchase amount
- **10 Customer Segments** identified with custom scoring system

### 2. Feature Engineering
Created safe features to avoid data leakage:
- `FM_Score`: Combined Frequency and Monetary scores
- `Monetary_per_Frequency`: Average transaction value
- Segment indicators for Champions, Loyal, Promising, New Customers

### 3. Model Development
Compared multiple algorithms with rigorous cross-validation:
- **Logistic Regression** (Selected): 86.1% ± 1.1% ROC-AUC
- **XGBoost**: 88.0% ROC-AUC (single test)
- **Random Forest**: 83.0% ± 1.2% ROC-AUC

## 🏆 Key Results

### Customer Segmentation
- **Champions (12.1%)**: £3.89M revenue contribution (53.3% of total)
- **At Risk (4.1%)**: £344K potential loss if not retained
- **Lost (27.6%)**: Already churned customers requiring win-back campaigns

### Model Performance
| Metric | Value | Business Impact |
|--------|-------|----------------|
| ROC-AUC | 0.861 | Excellent discrimination |
| Recall | 0.946 | Catches 94.6% of churners |
| Precision | 0.581 | 58.1% prediction accuracy |

### Business Impact
- **304 high-risk customers** identified
- **£180,953 potential revenue** at risk
- **£595 average** customer value
- **1,418% ROI** through targeted retention

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### Usage
```python
from src.rfm_analysis import calculate_rfm_scores
from src.model_utils import predict_churn

# Load and analyze customer data
rfm_scores = calculate_rfm_scores(transaction_data)

# Predict churn probability
churn_predictions = predict_churn(customer_features)
```

## 📊 Key Visualizations

### Customer Segmentation
![RFM Segments](visualizations/rfm_segments.png)

### Model Performance
![Model Comparison](visualizations/model_comparison.png)

## 🔧 Technical Highlights

### Data Leakage Resolution
Initially achieved suspiciously perfect 100% ROC-AUC, which led to investigation:
- **Problem**: Using Recency (days since last purchase) as both target definition and feature
- **Solution**: Removed all recency-related features and rebuilt model
- **Result**: More realistic but still excellent 86.1% performance

### Feature Importance Discovery
- **FM_Score (50.9%)**: Combined frequency and monetary value most predictive
- **New Customer Segment (25.0%)**: Segment membership highly informative
- **Individual scores**: Less predictive than combined features

## 📈 Business Recommendations

1. **Immediate Action**: Contact 304 high-risk customers with retention offers
2. **Segment Strategy**: Develop targeted campaigns for each customer segment
3. **Monitoring**: Implement monthly model retraining pipeline
4. **Expansion**: Extend model to predict Customer Lifetime Value (CLV)

## 🔬 Future Enhancements

- [ ] Real-time prediction API deployment
- [ ] Deep learning models (LSTM for sequential patterns)
- [ ] Customer Lifetime Value prediction
- [ ] A/B testing framework for retention campaigns
- [ ] Advanced ensemble methods

## 📚 Key Learnings

This project demonstrates critical data science skills:
- **Critical thinking**: Questioning perfect results led to data leakage discovery
- **Problem diagnosis**: Systematic approach to identifying and fixing issues
- **Business translation**: Converting technical metrics to actionable insights
- **End-to-end pipeline**: From raw data to business recommendations

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The online retail business community for domain knowledge
- Open source contributors for the amazing Python ecosystem

---

**Contact**: [Your Email] | [LinkedIn Profile] | [Portfolio Website]

*"Turning data into actionable customer insights"*
