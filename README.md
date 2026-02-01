# 🌱 Crop Recommendation System using Machine Learning & Deep Learning

### Submitted by: 
1. Divya JAYAPRAKASH
2. Jayasri DHANAPAL
3. Reshma KARTHIKEYAN NAIR

## Project Overview
Choosing the right crop based on soil nutrients and environmental conditions is a major challenge in agriculture. Traditional farming practices rely on experience and manual decision-making, which may not always lead to optimal crop yield.

This project presents a complete **Machine Learning (ML)** and **Deep Learning (DL)** pipeline applied to an agriculture-related problem: **crop recommendation based on soil nutrients and climatic conditions**.

The objective is not only to achieve high predictive performance, but also to demonstrate a **rigorous, interpretable, and reproducible data science workflow**. The project covers exploratory data analysis, feature engineering, feature selection, model comparison, hyperparameter tuning, explainable AI, deep learning implementation, and critical analysis.

---

## Problem Definition
- **Problem Type:** Multi-class Classification  
- **Goal:** Predict the most suitable crop given soil and climate conditions  
- **Target Variable:** Crop label  
- **Number of Classes:** 22 crops  

The system supports data-driven agricultural decision-making by replacing subjective judgment with quantitative analysis.

---

## Dataset Description
- **Dataset Name:** Crop Recommendation Dataset  
- **Source:** Public agricultural dataset  
- **Total Samples:** 2,200  
- **Class Distribution:** Perfectly balanced across 22 crop types  

### Input Features
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

### Dataset Characteristics
- No missing values  
- All features are numerical  
- Suitable for both ML and DL models  

---

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the dataset structure and statistical properties.

### Key Analyses
- Feature distribution analysis using boxplots  
- Outlier detection in soil and climate variables  
- Verification of target class balance  
- Preliminary inspection of feature relationships  

EDA confirmed that the dataset is clean, balanced, and suitable for supervised learning.

---

## Data Preprocessing
- Stratified train-test split to preserve class distribution  
- Feature standardization to ensure scale consistency  
- Fixed random states for reproducibility  

Preprocessing ensured fair comparison across models.

---

## Feature Engineering
Domain-driven feature engineering was applied to enhance model learning:

- Nutrient ratios: **N/P, N/K, P/K**
- Soil fertility index derived from N, P, and K
- Climate interaction feature combining temperature, humidity, and rainfall

These features significantly improved predictive performance.

---

## Feature Selection
- Correlation analysis to detect redundant features  
- Random Forest feature importance ranking  

Feature selection improved interpretability without sacrificing accuracy.

---

## Machine Learning Models Evaluated
The following ML models were implemented and compared:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

Each model represents a different learning paradigm, enabling comprehensive evaluation.

---

## Hyperparameter Tuning
- **GridSearchCV** applied to Random Forest  
- **RandomizedSearchCV** applied to SVM  
- Cross-validation used for robust performance estimation  

Performance before and after tuning was compared to validate improvements.

---

## Explainability (SHAP)
To address the black-box nature of predictive models,Tto improve transparency and interpretability **SHAP (SHapley Additive exPlanations)** was used.

- Identified key features influencing crop prediction
- Provided both global and local explanations  
- Analysis of how soil nutrients and climate variables influence predictions  

This improves trust in the model’s decision-making process.

---

## Deep Learning Model
A Deep Learning model using **Artificial Neural Networks (ANN)** was developed for tabular data:

- Input layer representing soil and climate features
- Hidden layers to capture complex relationships
- Output layer representing crop classes
- Fully connected dense layers  
- Dropout regularization to prevent overfitting  
- Early stopping based on validation loss  

The DL model demonstrated competitive performance compared to traditional ML models.

---

## Model Performance Comparison

| Model               | Accuracy |
|--------------------|----------|
| Random Forest       | 98.86%   |
| Decision Tree       | 98.18%   |
| Logistic Regression | 97.73%   |
| ANN                 | 97.50%   |
| KNN                 | 96.36%   |
| SVM                 | 96.36%   |

Random Forest achieved the best balance between accuracy and interpretability.

---

## Key Insights
- Tree-based models perform best on structured agricultural data  
- Feature engineering plays a crucial role in improving performance  
- ANN is effective but not superior for this dataset  
- SHAP highlights climate and soil fertility as dominant factors  
- Interpretability is essential for agricultural decision support systems  

---

## Limitations
- Dataset is static and does not capture seasonal variations  
- Geographic generalization may be limited  
- Real-world agricultural data may include higher noise  
- Deep learning interpretability remains limited  

---

## Future Work
- Incorporate time-series weather data 
- Integrate satellite or remote sensing information  
- Expand dataset geographically  
- Deploy as a farmer decision-support system  

---

## Technologies Used
- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- SHAP  

---

## Project Structure
├── Data/ 
│   └── Crop_recommendation.csv 
├── Notebooks/ 
│   └── Crop_Recommendation_ML_DL.ipynb 
├── README.md 
├── Crop Recommendation Report.pdf