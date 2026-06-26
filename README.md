# M516 Business Project – Housing Rental Price Prediction in Germany

## Overview
This project predicts monthly rental prices across German metropolitan areas using machine learning. Two models are developed and compared: a Linear Regression baseline and a tuned XGBoost model, trained on 370,000+ rental listings from ImmoScout24.

## Repository Structure
```
├── M516_FINAL.ipynb       # Main notebook (EDA, modelling, results)
├── immo_data.csv          # Dataset (ImmoScout24 rental listings)
├── outputs/               # Generated plots and predictions
│   ├── 01_correlations.png
│   ├── 02_distributions.png
│   ├── 03_model_comparison.png
│   ├── 04_predictions_analysis.png
│   ├── 05_feature_importance.png
│   └── predictions.csv
└── README.md
```

## Dataset(
- **Source:** ImmoScout24
- **Size:** 370,000+ rental listings across 1,000+ German cities
- **Features used:** `livingSpace`, `noRooms`, `yearConstructed`, `hasKitchen`, `balcony`, `regio1`, `regio2`, `baseRent`
- link :https://drive.google.com/drive/folders/1unsYTXdet0MjTCuMPQKcpIe-6bMu2vsD?usp=sharing

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/M516-Rental-Price-Prediction
   ```
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Place `immo_data.csv` in the root directory
4. Open and run `M516_FINAL.ipynb` in Jupyter Notebook or JupyterLab

## Results Summary

| Metric | Linear Regression | XGBoost Tuned |
|---|---|---|
| MAE (€/month) | €143 | €104 |
| RMSE (€/month) | €224 | €176 |
| R² Score | 0.775 | 0.860 |

XGBoost outperforms the baseline by 27.3% in MAE and explains 86% of rental price variance.

## Key Findings
- `livingSpace` is the dominant predictor (55.4% feature importance)
- `city_encoded` captures the geographic location premium (28.2%)
- Munich commands the highest predicted rent at ~€1,696/month
- Non-linear interactions between features cannot be captured by linear models alone

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
