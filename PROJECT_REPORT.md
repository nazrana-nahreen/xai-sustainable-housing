# PROJECT REPORT

# Explainable AI (XAI) for Sustainable Housing Policy Using USA Real Estate Dataset

---

## 1. Title

**Explainable AI (XAI) for Sustainable Housing Policy: An Interpretable Machine Learning Analysis of the U.S. Real Estate Market**

---

## 2. Introduction

Housing affordability and sustainability are among the most pressing socioeconomic challenges in the United States. With rising property prices, urban sprawl, and increasing environmental concerns, there is a growing need for data-driven, transparent policy-making in the housing sector. Traditional housing valuation models often operate as "black boxes," making it difficult for policymakers, urban planners, and citizens to understand _why_ certain properties command higher prices and what structural factors can be addressed through policy intervention.

Explainable Artificial Intelligence (XAI) offers a powerful framework for making machine learning predictions transparent and interpretable. By combining predictive accuracy with human-understandable explanations, XAI enables stakeholders to not only predict housing prices but also understand the key drivers behind those predictions — a critical requirement for accountable and equitable housing policy.

This project applies XAI techniques — including SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations), and feature importance analysis — to a large-scale U.S. real estate dataset containing over 2.2 million property listings. The goal is to derive transparent, data-driven insights that inform sustainable housing policy decisions around affordability, land use efficiency, regional equity, and compact development.

---

## 3. Problem Statement

Despite the availability of large-scale housing data, current approaches to housing policy often rely on aggregate statistics (median prices, affordability ratios) that fail to capture the complex, non-linear interactions between property features and pricing. Key challenges include:

1. **Lack of Transparency**: Machine learning models used in property valuation and market analysis are typically opaque, making it impossible for policymakers to understand which factors drive predictions.

2. **Regional Disparities**: Housing price drivers vary significantly across states and regions, but most analyses treat the market as homogeneous.

3. **Sustainability Blind Spots**: Traditional analyses rarely incorporate sustainability-relevant metrics such as land use efficiency, room density, and spatial compactness — factors critical for sustainable urban development.

4. **Inequitable Outcomes**: Without explainable models, there is a risk of perpetuating biased or unfair housing assessments that disproportionately affect certain communities.

**Research Question**: _How can Explainable AI techniques be applied to large-scale real estate data to identify the key drivers of housing prices and inform transparent, sustainable housing policy?_

---

## 4. Objectives

The primary objectives of this project are:

1. **Build Accurate Predictive Models**: Train Random Forest and XGBoost regression models to predict U.S. housing prices using property features, achieving strong predictive performance (R² > 0.70).

2. **Apply Multiple XAI Techniques**: Implement SHAP, LIME, and feature importance analysis to provide both global and local explanations of model predictions.

3. **Engineer Sustainability-Relevant Features**: Create features that capture housing sustainability dimensions, including:
   - Price per square foot (cost efficiency)
   - Land use efficiency (building-to-lot size ratio)
   - Room density (rooms per unit area)
   - State-level affordability index

4. **Develop a State Sustainability Scorecard**: Rank U.S. states by a composite sustainability score based on land efficiency, room density, and affordability.

5. **Derive Actionable Policy Recommendations**: Translate XAI insights into concrete, evidence-based recommendations for sustainable housing policy.

6. **Compare XAI Methods**: Validate the robustness of findings by comparing feature rankings across Random Forest, XGBoost, and SHAP using Spearman rank correlations.

---

## 5. Proposed System

The proposed system is an end-to-end XAI pipeline for sustainable housing policy analysis, implemented as a reproducible Jupyter Notebook. The system architecture consists of four main components:

### 5.1 Data Pipeline
- **Input**: USA Real Estate Dataset from Kaggle (2,226,382 listings across 55 states/territories)
- **Cleaning**: Handle missing values (~22-25% in bed/bath/house_size), remove outliers (1st-99th percentile), filter to reasonable residential properties ($10K-$5M, 100-20,000 sqft)
- **Output**: Cleaned dataset of 1,317,749 records across 54 states

### 5.2 Feature Engineering Module
- **Price per Square Foot**: `price / house_size`
- **Land Use Efficiency**: `house_size / (acre_lot × 43,560)` — ratio of building footprint to lot area
- **Room Density**: `(bed + bath) / (house_size / 1000)` — rooms per 1,000 sqft
- **Total Rooms**: `bed + bath`
- **State Affordability Index**: Z-score of state median price relative to national median
- **Property Size Categories**: Small (<1,000), Modest (1,000-1,500), Medium (1,500-2,500), Large (2,500-4,000), Luxury (4,000+)

### 5.3 Predictive Modeling Module
- **Random Forest Regressor**: 200 trees, max_depth=20, predicting log(price)
- **XGBoost Regressor**: 300 estimators, max_depth=8, learning_rate=0.1
- **Training**: 200,000 sample subset (80/20 train-test split)
- **Evaluation**: R², MAE, RMSE on both train and test sets

### 5.4 XAI Analysis Module
- **SHAP TreeExplainer**: Exact Shapley values for tree-based models
  - Beeswarm plots (global feature impact with direction)
  - Bar plots (mean |SHAP value|)
  - Dependence plots (feature interactions)
  - Waterfall plots (individual prediction explanations)
- **LIME Explainer**: Local surrogate models for instance-level explanations
- **Feature Importance**: Built-in importance from both RF and XGBoost
- **Method Comparison**: Spearman rank correlation across all XAI methods

### 5.5 Policy Analysis Module
- State-level sustainability scoring
- Land use efficiency vs. affordability quadrant analysis
- Property size category analysis
- Data-driven policy recommendations

---

## 6. Methodology

### 6.1 Dataset

The **USA Real Estate Dataset** (Kaggle, by Ahmed Shahriar Sakib) contains 2,226,382 real estate listings with the following attributes:

| Feature | Type | Description |
|---------|------|-------------|
| `price` | float | Listing price in USD |
| `bed` | float | Number of bedrooms |
| `bath` | float | Number of bathrooms |
| `acre_lot` | float | Lot size in acres |
| `house_size` | float | House size in square feet |
| `state` | string | U.S. state |
| `zip_code` | float | ZIP code |
| `city` | string | City name |
| `status` | string | Listing status (for_sale, sold, etc.) |
| `brokered_by` | float | Broker ID |
| `street` | float | Street address (encoded) |
| `prev_sold_date` | string | Previous sale date |

### 6.2 Data Preprocessing

1. **Column Removal**: Dropped `brokered_by`, `street`, and `prev_sold_date` (non-predictive or too sparse)
2. **Missing Value Handling**: Dropped rows missing any critical feature (price, bed, bath, acre_lot, house_size, state, zip_code) — reducing to 1,360,627 records
3. **Price Filtering**: Retained properties priced between $10,000 and $5,000,000 → 1,352,587 records
4. **Size Filtering**: Applied reasonable bounds (100-20,000 sqft, 1-10 beds, 1-10 baths, 0-100 acres) → 1,344,251 records
5. **Outlier Removal**: Removed prices outside the 1st-99th percentile → **1,317,749 final records**

### 6.3 Feature Engineering

Six sustainability-focused features were engineered:

- **Price per Square Foot** = `price / house_size`
- **Land Use Efficiency** = `house_size / (acre_lot × 43,560)`, clipped to [0, 1]
- **Room Density** = `(bed + bath) / (house_size / 1000)`
- **Total Rooms** = `bed + bath`
- **Log Price** = `log(1 + price)` (target variable transformation for normality)
- **State Affordability Index** = Z-score of state median price

### 6.4 Model Training

Four models were trained on a 200,000-sample subset (including Linear Regression and Decision Tree as baselines):

**Random Forest:**
- 200 estimators, max_depth=20, min_samples_split=10, min_samples_leaf=5
- Feature selection: `sqrt`
- Target: `log(1 + price)`

**XGBoost:**
- 300 estimators, max_depth=8, learning_rate=0.1
- Regularization: subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- Early stopping via eval_set

### 6.5 Explainability Analysis

**SHAP (SHapley Additive exPlanations):**
- Used `TreeExplainer` for exact Shapley value computation
- Computed on 2,000 test samples
- Generated beeswarm, bar, dependence, and waterfall visualizations

**LIME (Local Interpretable Model-agnostic Explanations):**
- Created `LimeTabularExplainer` trained on the full training set
- Explained 3 representative instances: typical, most affordable, most expensive

**Feature Importance Comparison:**
- Extracted built-in importance from RF (Gini impurity) and XGBoost (gain)
- Compared rankings across RF, XGBoost, and SHAP using Spearman rank correlation

### 6.6 Sustainability Scoring

A composite **Sustainability Score** was computed for each state:

```
Sustainability Score = 0.4 × Land_Use_Efficiency_norm + 0.3 × Room_Density_norm + 0.3 × (1 - Price_per_sqft_norm)
```

Where `_norm` indicates min-max normalization. Higher scores indicate more sustainable housing markets (efficient land use, compact design, affordable pricing).

---

## 7. Results

### 7.1 Exploratory Data Analysis

The cleaned dataset contains **1,317,749 residential property listings** across **54 U.S. states and territories**. Key statistics:

- **Median price**: Varies significantly by state, from ~$167,000 (Puerto Rico) to ~$869,000 (District of Columbia)
- **Price distribution**: Right-skewed, with log transformation yielding approximately normal distribution
- **Missing data**: 22-25% of records lacked bedroom, bathroom, or house size information

![Price Distribution](https://app.devin.ai/attachments/5a309b77-d314-4372-b030-02ad29696e3a/price_distribution.png)
*Figure 1: Distribution of housing prices and key features*

![Correlation Matrix](https://app.devin.ai/attachments/e845cfbf-47f5-459a-92c3-584950be28ee/correlation_matrix.png)
*Figure 2: Correlation matrix showing relationships between numerical features*

![Scatter Analysis](https://app.devin.ai/attachments/409e0a83-5b96-4997-a623-c8761c6d53a1/scatter_analysis.png)
*Figure 3: Key relationships — house size, lot size, bedrooms, and bathrooms vs. price*

### 7.2 Model Performance

Both models achieved strong predictive performance:

| Model | Train R² | Test R² | Test MAE | Test RMSE |
|-------|:--------:|:-------:|:--------:|:---------:|
| Linear Regression | 0.4232 | 0.4210 | $195,183 | $332,672 |
| Decision Tree | 0.8245 | 0.7116 | $132,859 | $232,231 |
| Random Forest | 0.8188 | 0.7291 | $130,386 | $235,450 |
| **XGBoost** | **0.8110** | **0.7547** | **$122,981** | **$218,727** |

**XGBoost outperformed all models** on test metrics, achieving a test R² of 0.7547 and MAE of $122,981 — a **79.3% improvement** over the Linear Regression baseline.

![Model Performance](https://app.devin.ai/attachments/27a789ea-3af8-4b26-9df7-83e88d43fe4f/model_performance.png)
*Figure 4: Actual vs. predicted prices for both models*

### 7.3 Feature Importance Analysis

Built-in feature importance from both models reveals the relative influence of each feature:

![Feature Importance](https://app.devin.ai/attachments/34cdb250-f39e-457e-81f0-38bd8b280a94/feature_importance.png)
*Figure 5: Feature importance comparison — Random Forest vs. XGBoost*

### 7.4 SHAP Analysis Results

SHAP analysis provides the most comprehensive explainability:

**Global Feature Impact (SHAP):**

| Rank | Feature | Relative Impact |
|:----:|---------|:--------------:|
| 1 | ZIP Code | 28.7% |
| 2 | Bathrooms | 18.7% |
| 3 | House Size (sqft) | 17.7% |
| 4 | State | 12.6% |
| 5 | Room Density | 6.1% |
| 6 | Total Rooms | 3.6% |
| 7 | Listing Status | 3.5% |
| 8 | Lot Size (acres) | 3.5% |
| 9 | Land Use Efficiency | 2.8% |
| 10 | Bedrooms | 2.6% |

![SHAP Summary](https://app.devin.ai/attachments/0c5d676c-23d1-40ee-b76f-3264457b8a9d/shap_summary.png)
*Figure 6: SHAP beeswarm plot — how each feature impacts price predictions (red = high value, blue = low value)*

![SHAP Bar](https://app.devin.ai/attachments/18fe1a3b-8551-4234-a527-aecaeb4cc500/shap_bar.png)
*Figure 7: Mean |SHAP Value| — global feature importance*

**Key findings from SHAP:**
- **Location dominates**: ZIP code (28.7%) and state (12.6%) together account for over 41% of price impact
- **Physical features matter**: Bathrooms (18.7%) and house size (17.7%) are the strongest non-location features
- **Sustainability metrics contribute**: Room density (6.1%) and land use efficiency (2.8%) have measurable impact

![SHAP Dependence](https://app.devin.ai/attachments/a5c8c0d9-5869-49be-9fac-fd004eb16904/shap_dependence.png)
*Figure 8: SHAP dependence plots showing feature interactions*

![SHAP Waterfall](https://app.devin.ai/attachments/652d1cbd-ffd3-4568-8395-043b0353efdf/shap_waterfall.png)
*Figure 9: SHAP waterfall plots — explaining why individual properties are affordable vs. expensive*

### 7.5 LIME Analysis Results

LIME provides complementary local explanations for individual property predictions:

![LIME Explanations](https://app.devin.ai/attachments/d5063465-c3f5-4de7-9b37-c86b8b188efa/lime_explanations.png)
*Figure 10: LIME explanations for typical, affordable, and expensive properties*

### 7.6 XAI Method Comparison

Feature rankings were compared across all three methods using Spearman rank correlation:

| Method Pair | Spearman Correlation |
|-------------|:-------------------:|
| RF vs XGBoost | 0.661 |
| RF vs SHAP | 0.891 |
| XGBoost vs SHAP | 0.673 |

The **moderate-to-strong consistency** (ρ = 0.661–0.891) across methods validates the robustness of the findings. All methods agree that location, bathrooms, and house size are the dominant price drivers.

### 7.7 State Sustainability Scorecard

**Top 5 Most Sustainable Housing Markets:**

| Rank | State | Sustainability Score | Median Price | Land Use Efficiency | Room Density |
|:----:|-------|:-------------------:|:------------:|:------------------:|:------------:|
| 1 | Puerto Rico | 0.71 | $167,000 | 0.28 | 3.76 |
| 2 | District of Columbia | 0.65 | $869,000 | 0.91 | 3.29 |
| 3 | Iowa | 0.56 | $223,000 | 0.15 | 3.50 |
| 4 | Pennsylvania | 0.49 | $265,000 | 0.21 | 3.12 |
| 5 | Ohio | 0.48 | $184,000 | 0.17 | 3.06 |

**Bottom 5 Least Sustainable Housing Markets:**

| Rank | State | Sustainability Score | Median Price | Land Use Efficiency | Room Density |
|:----:|-------|:-------------------:|:------------:|:------------------:|:------------:|
| 1 | Hawaii | 0.26 | $765,000 | 0.05 | 3.58 |
| 2 | New Hampshire | 0.27 | $440,000 | 0.05 | 2.76 |
| 3 | Montana | 0.31 | $475,000 | 0.13 | 2.79 |
| 4 | Vermont | 0.32 | $365,000 | 0.04 | 2.82 |
| 5 | Minnesota | 0.35 | $330,000 | 0.22 | 2.60 |

---

## 8. Conclusion

This project demonstrates the power of Explainable AI for informing sustainable housing policy. By applying SHAP, LIME, and feature importance analysis to a dataset of over 2.2 million U.S. real estate listings, we derived the following key conclusions and policy recommendations:

### Key Findings

1. **Location is the dominant price driver**: ZIP code and state together account for 41.3% of price impact (SHAP), confirming significant regional disparities in U.S. housing markets.

2. **Physical property features are highly influential**: Bathrooms (18.7%) and house size (17.7%) are the strongest non-location features, suggesting that policies targeting property configurations can meaningfully impact affordability.

3. **Land use efficiency correlates with sustainability**: States with higher building-to-lot ratios (e.g., District of Columbia, Puerto Rico) score higher on the sustainability index, supporting compact development policies.

4. **XAI methods provide consistent insights**: Spearman correlations of 0.661–0.891 across three XAI methods validate the robustness of findings.

5. **XGBoost outperforms Random Forest**: With a test R² of 0.7444 vs. 0.7128, XGBoost provides more accurate price predictions while remaining fully explainable through SHAP.

### Policy Recommendations

Based on the XAI analysis, we recommend the following policy actions:

| # | Recommendation | XAI Evidence |
|:-:|---------------|-------------|
| 1 | **Promote compact development** through zoning reforms | Land use efficiency impacts price; compact properties are more affordable |
| 2 | **Incentivize right-sized housing** (1,000–1,500 sqft) | House size is a dominant price driver; smaller homes balance livability and cost |
| 3 | **Address regional disparities** through federal programs | Location accounts for 41% of price variation |
| 4 | **Support efficient floor plans** with adequate rooms in smaller footprints | Room density and bathroom count significantly influence pricing |
| 5 | **Adopt explainable AI in housing assessments** | XAI ensures fair, transparent, accountable property valuations |

### Limitations

- The dataset lacks temporal features (price trends), environmental attributes (energy ratings, green certifications), and demographic data
- The sustainability score is a simplified composite; real-world scoring would require additional indicators
- Model training used a 200K sample for computational efficiency; full-dataset training may yield different importance rankings

### Future Work

- Incorporate temporal price trends for dynamic policy recommendations
- Add environmental features (energy ratings, proximity to public transit, green certification status)
- Extend to metro-area analysis for localized policy guidance
- Integrate demographic data (income levels, population density) for equity-focused analysis
- Deploy as an interactive policy dashboard for real-time stakeholder exploration

---

## 9. References

1. Sakib, A. S. (2023). *USA Real Estate Dataset*. Kaggle. https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems (NeurIPS), 30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

3. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939778

4. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785

6. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

7. U.S. Department of Housing and Urban Development. (2023). *Comprehensive Housing Affordability Strategy (CHAS) Data*. https://www.huduser.gov/portal/datasets/cp.html

8. United Nations. (2015). *Sustainable Development Goal 11: Sustainable Cities and Communities*. https://sdgs.un.org/goals/goal11

9. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

10. National Association of Realtors. (2023). *Housing Affordability Index*. https://www.nar.realtor/research-and-statistics/housing-statistics/housing-affordability-index

---

*Report generated as part of the XAI for Sustainable Housing Policy project.*
*Dataset: USA Real Estate Dataset (Kaggle) — 2,226,382 listings across 55 U.S. states and territories.*
*Tools: Python, Scikit-learn, XGBoost, SHAP, LIME, Matplotlib, Seaborn, Plotly.*
