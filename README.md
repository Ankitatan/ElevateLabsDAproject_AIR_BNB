# 🏡 Airbnb Dynamic Pricing Recommendation Engine

A machine learning project that predicts Airbnb nightly listing prices using property characteristics and presents interactive pricing scenarios through a Tableau dashboard.

## 📌 Project Overview

Airbnb hosts need to set competitive nightly prices based on location, room type, availability, and listing characteristics. This project explores how machine learning can estimate listing prices and help users compare alternative pricing scenarios.

The project combines **Python, Machine Learning, Data Analysis, and Tableau** to build a data-driven pricing analysis workflow.

## 🎯 Objectives

- Analyze Airbnb listing data to identify pricing patterns.
- Clean and preprocess data for machine learning.
- Build a regression model to predict nightly listing prices.
- Evaluate model performance using standard regression metrics.
- Visualize pricing patterns using Tableau.
- Create an interactive price-adjustment slider for scenario analysis.

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python | Data processing and model development |
| Pandas, NumPy | Data cleaning and manipulation |
| Matplotlib, Seaborn | Exploratory data analysis and visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Random Forest Regressor | Nightly price prediction |
| Tableau | Interactive dashboard and price scenarios |
| Google Colab | Python development environment |

## 📂 Dataset

The project uses Airbnb listing data containing fields such as:

- `neighbourhood_group` and `neighbourhood`
- `latitude` and `longitude`
- `room_type`
- `price`
- `minimum_nights`
- `number_of_reviews`
- `reviews_per_month`
- `calculated_host_listings_count`
- `availability_365`

The dataset provides listing-level information. It does not contain a complete historical daily calendar of nightly prices, so the project does not directly model actual seasonal price changes.

## ⚙️ Methodology

1. **Data cleaning:** Inspect missing values, remove duplicate records where appropriate, and handle invalid or unsuitable price values.
2. **Exploratory data analysis:** Examine price distributions and differences by neighbourhood and room type.
3. **Feature engineering:** Select relevant numerical and categorical listing features.
4. **Preprocessing:** Encode categorical features and prepare data for model training.
5. **Model training:** Train a Random Forest Regression model using a training dataset.
6. **Model evaluation:** Evaluate predictions on a held-out test set.
7. **Price estimation:** Generate estimated nightly prices for listings.
8. **Tableau visualization:** Build interactive visualizations and a percentage-based price-adjustment slider.

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Regressor

This supervised learning algorithm combines predictions from multiple decision trees to estimate a continuous numerical target: the listing's nightly price.

### Features used

The model uses listing attributes such as location, room type, geographic coordinates, minimum nights, review activity, host listing count, and availability.

### Target variable

`price`

### Model evaluation results

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | $65.46 |
| Root Mean Squared Error (RMSE) | $184.35 |
| R² Score | 0.151 |

These are the reported test-set results from the current model.

The R² score indicates limited predictive performance. The MAE and RMSE also show that predictions can differ substantially from actual listing prices. These results motivate further feature engineering, outlier analysis, cross-validation, and hyperparameter tuning.

## 📊 Tableau Dashboard

The interactive dashboard includes six worksheets and one dashboard, with visualizations covering listing prices, neighbourhoods, room types, and other available listing attributes.

### Key dashboard features

- Price analysis by neighbourhood.
- Price comparison by room type.
- Listing distribution by area.
- Additional exploratory visualizations.
- Predicted nightly price analysis.
- Interactive price-adjustment slider.

### Price-adjustment slider

The slider allows users to explore pricing scenarios by adjusting the model's predicted price:

- **80%:** 20% below the predicted price.
- **100%:** The original predicted price.
- **120%:** 20% above the predicted price.

The slider is a scenario-analysis control. It does not retrain the model or determine the price that maximizes bookings or revenue.

> **Dashboard preview:** Add your Tableau dashboard screenshot here after uploading it to the repository. For example, save it as `images/airbnb_dashboard.png` and uncomment the line below.
>
> <!-- ![Airbnb Dynamic Pricing Tableau Dashboard](images/airbnb_dashboard.png) -->

## 📁 Project Structure

```text
Airbnb-Dynamic-Pricing/
│
├── README.md
├── AB_NYC_2019.csv
├── airbnb_cleaned.csv
├── airbnb_tableau_predictions.csv
├── airbnb_pricing_model.ipynb
├── images/
│   └── airbnb_dashboard.png
└── tableau/
    └── airbnb_dashboard.twbx
```

*This is a suggested structure. Adjust the filenames and folders to match the files actually present in your repository. Do not upload large datasets or generated files unless you have the right to share them.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ankitatan/Airbnb-Dynamic-Pricing.git
cd Airbnb-Dynamic-Pricing
```

Replace the repository URL with your actual GitHub repository URL if the repository has a different name.

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order. Make sure the dataset path in the notebook matches the location of your CSV file.

### 4. Open the Tableau dashboard

Open the packaged Tableau workbook (`.twbx`) if available. If using a `.twb` workbook, ensure the required data files are accessible and reconnect any missing data sources.

## 💡 Business Applications

This project demonstrates how data analytics and machine learning can support:

- Comparing estimated prices across neighbourhoods.
- Exploring differences between room types.
- Reviewing how listing characteristics relate to price.
- Evaluating alternative price scenarios.
- Communicating analytical findings through an interactive dashboard.

These are potential applications; the project has not established that its suggested prices increase occupancy or revenue.

## 🔮 Future Improvements

- Improve predictive performance through feature engineering and hyperparameter tuning.
- Compare Random Forest with Linear Regression, Gradient Boosting, and other regression algorithms.
- Investigate outliers and prediction errors across different price ranges.
- Add richer property attributes where reliable data is available.
- Incorporate historical calendar prices to study seasonality.
- Evaluate pricing recommendations using booking and revenue outcomes when those data become available.
- Deploy an interactive prediction application using Streamlit.

## 📚 Skills Demonstrated

Python · Data Cleaning · Exploratory Data Analysis · Feature Engineering · Supervised Machine Learning · Regression · Model Evaluation · Tableau · Data Visualization · Business Insights

## 👩‍💻 Author

**Ankita Taneja**

Aspiring Data Analyst | Python | SQL | Machine Learning | Tableau | Power BI

- GitHub: [Ankitatan](https://github.com/Ankitatan)
- LinkedIn: [Ankita Taneja](https://www.linkedin.com/in/ankita-t-390613396/)

## ⚠️ Disclaimer

This is an analytical and learning project. The model provides estimated listing prices based on the available dataset and has limited predictive performance. Its outputs should not be interpreted as guaranteed optimal prices or financial advice.

## 📄 License

No license has been specified yet. Add an appropriate `LICENSE` file if you intend to permit reuse or redistribution.
