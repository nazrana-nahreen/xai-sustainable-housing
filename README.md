# Explainable AI (XAI) for Sustainable Housing Policy

An analysis leveraging Explainable AI techniques to derive actionable insights for sustainable housing policy using the [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset).

## Overview

This project applies machine learning and Explainable AI (XAI) methods to understand the key factors driving housing prices across the United States. By making model predictions transparent and interpretable, we aim to inform sustainable housing policy decisions around affordability, land use efficiency, and equitable development.

## Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of 2.2M+ real estate listings across all US states
- **Sustainability-Focused Feature Engineering**: Price per square foot, land use efficiency, density metrics, and affordability indices
- **Predictive Modeling**: Random Forest and XGBoost models for housing price prediction
- **Explainable AI (XAI)**:
  - **SHAP (SHapley Additive exPlanations)**: Global and local feature importance, interaction effects, and dependence plots
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Instance-level explanations for individual predictions
  - **Feature Importance**: Permutation-based and built-in feature importance analysis
- **Policy Insights**: Data-driven recommendations for sustainable housing policy

## Project Structure

```
xai-sustainable-housing/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                          # Dataset (not tracked in git)
│   └── realtor-data.csv
└── notebooks/
    └── xai_sustainable_housing.ipynb  # Main analysis notebook
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd xai-sustainable-housing

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the dataset from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("ahmedshahriarsakib/usa-real-estate-dataset")
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) and place the CSV file in the `data/` directory.

### Running the Analysis

```bash
jupyter notebook notebooks/xai_sustainable_housing.ipynb
```

## Methodology

### 1. Data Preprocessing
- Handle missing values and outliers
- Filter to relevant property types and price ranges
- Create train/test splits stratified by state

### 2. Feature Engineering
- **Price per Square Foot**: Housing cost efficiency metric
- **Land Use Efficiency**: Ratio of house size to lot size
- **Room Density**: Rooms per square foot
- **Affordability Index**: State-level affordability classification
- **Property Size Categories**: Small, medium, large, luxury classifications

### 3. Model Training
- **Random Forest**: Ensemble method with strong interpretability
- **XGBoost**: Gradient boosting for high predictive accuracy
- Hyperparameter tuning via cross-validation

### 4. Explainability Analysis
- **Global Explanations**: Which features matter most across all predictions?
- **Local Explanations**: Why was a specific property priced this way?
- **Interaction Effects**: How do features interact to influence price?
- **Policy-Relevant Insights**: What can policymakers learn from the model?

## Policy Implications

The XAI analysis reveals key factors for sustainable housing policy:

1. **Land Use Efficiency**: How lot size vs. house size ratios affect affordability
2. **Regional Disparities**: State-level differences in housing price drivers
3. **Density vs. Affordability**: Trade-offs between housing density and cost
4. **Property Size Optimization**: Optimal property configurations for affordable housing

## License

This project is for educational and research purposes.

## Dataset Citation

Ahmed Shahriar Sakib. (2023). USA Real Estate Dataset. Kaggle.
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
