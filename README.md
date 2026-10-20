# 🏠 Airbnb Dynamic Pricing Recommendation Engine

## Overview
The Airbnb Dynamic Pricing Recommendation Engine is a machine learning project that predicts estimated nightly prices for Airbnb listings based on location, room type, reviews, minimum nights, host listings, and availability.

The project uses a trained Random Forest regression model and an interactive Streamlit application to support data-driven pricing decisions.

## Objectives
- Analyze Airbnb listing data and pricing patterns.
- Identify important features influencing listing prices.
- Predict estimated nightly prices using machine learning.
- Provide an interactive interface for price recommendations.
- Visualize listing insights across neighbourhoods and room types.

## Tools and Technologies
- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Random Forest Regression
- **Model Storage:** Joblib
- **Web Application:** Streamlit
- **Visualization:** Streamlit charts, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab, Visual Studio Code

## Project Structure

```text
Project AIR_BNB/
├── app.py
├── airbnb_eda_cleaned.csv
├── airbnb_price_model.joblib
├── requirements.txt
├── README.md
├── Airbnb_Dynamic_Pricing_Report.pdf
└── airbnb_model_output/
    ├── model_metrics.txt
    └── airbnb_test_predictions.csv
```

## Features
- Interactive listing input fields
- Estimated nightly price prediction
- Price and listing-count charts by room type
- Listing location visualization
- Model evaluation metrics
- Download options for cleaned data and available test predictions

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Ankitatan/Airbnb-Dynamic-Pricing-Engine.git
cd Airbnb-Dynamic-Pricing-Engine
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run app.py
```

Ensure that `airbnb_eda_cleaned.csv` and `airbnb_price_model.joblib` are in the same directory as `app.py`.

## Model Evaluation
The project includes evaluation metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R², where available. Refer to `airbnb_model_output/model_metrics.txt` for the recorded results.

## Disclaimer
Predicted prices are estimates based on historical data and model performance. They are not guaranteed market prices and may not reflect current demand or real-time market conditions.

## Author
**Ankita Taneja**

GitHub: [@Ankitatan](https://github.com/Ankitatan)
