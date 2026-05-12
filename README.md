# Titanic Survival Prediction

## Project Overview

This project builds an end-to-end machine learning workflow to predict passenger survival from the Titanic dataset. The goal was not only to train a classification model, but to evaluate how well different models generalized, how reliable their probability estimates were, and where each model made mistakes.

The project follows a structured data science workflow:

1. Exploratory Data Analysis
2. Feature engineering and feature testing
3. Hypothesis testing and feature selection
4. Model buildout and hyperparameter tuning
5. Model evaluation and final model selection
6. Final model artifact preparation

The final selected model was **Logistic Regression** because it provided competitive performance, stronger interpretability, more stable calibration behavior, and clearer error patterns compared with the Support Vector Machine model.

---

## Business / Analytical Objective

The objective of this project was to predict whether a Titanic passenger survived using demographic, ticket, cabin, fare, and travel-related features.

Beyond prediction accuracy, the project focused on answering deeper model evaluation questions:

- Which features provided useful predictive signal?
- Did engineered features improve performance or introduce noise?
- Did tuned models generalize well to unseen data?
- Were predicted probabilities reliable enough for threshold tuning?
- Which classification threshold produced the best precision-recall balance?
- Where did each model make the most common errors?

---

## Repository Structure

```text
Workflow/
├── Data/
│   ├── raw_data.csv
│   ├── uncut_feat.csv
│   ├── significant_feat.csv
│   ├── linear.csv
│   ├── non_linear.csv
│   └── target.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Testing.ipynb
│   ├── Hypothesis_Testing.ipynb
│   ├── Modeling.ipynb
│   └── Model_evaluation.ipynb
│
├── Models/
│   ├── final_logistic_regression_model.pkl
│   └── final_model_metadata.json
│
└── Scripts/
    └── train_final_model.py
```

Note: The `Models/` and `Scripts/` folders represent the recommended final production structure for saving the trained model and related metadata.

---

## Notebook Workflow

### 1. Exploratory Data Analysis

The EDA notebook reviews the raw Titanic dataset to understand data quality, missing values, data types, duplicate records, distributions, outliers, and early feature relationships.

Key steps included:

- Checked dataset shape, duplicates, and data types
- Reviewed missing values
- Filled missing cabin values with `Unknown`
- Imputed missing age values using the mean
- Reviewed numeric summary statistics
- Visualized `Age` and `Fare` distributions
- Identified skewness and outliers in `Fare`
- Applied log transformation to reduce fare skewness
- Encoded categorical variables
- Scaled numeric variables for downstream modeling
- Created interaction and ratio features

The final EDA output was saved as an engineered feature dataset for later feature testing.

---

### 2. Feature Testing

The feature testing notebook evaluates whether engineered features added useful predictive signal or introduced noise.

Predictive Power Score was used to compare engineered interaction and ratio features against their original component variables. This helped determine whether each engineered feature improved predictive strength or caused informational dilution.

This phase helped identify which engineered features should be carried forward into hypothesis testing and modeling.

---

### 3. Hypothesis Testing and Feature Selection

The hypothesis testing notebook evaluates feature significance and multicollinearity.

Key methods included:

- Sequential Feature Selection for K-Nearest Neighbors
- Logistic Regression significance testing using coefficients, standard errors, z-scores, and p-values
- Variance Inflation Factor review for multicollinearity
- Removal of redundant or statistically weak variables
- Creation of separate curated datasets for linear and non-linear models

This phase produced:

- `linear.csv` for Logistic Regression
- `non_linear.csv` for SVM and other non-linear models
- `target.csv` for the survival target

---

### 4. Modeling

The modeling notebook tested several classification algorithms to identify strong candidate models.

Models tested:

- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest
- XGBoost

Logistic Regression and SVM produced the strongest candidate results and were selected for deeper evaluation.

Important note: GridSearchCV and HalvingGridSearchCV outputs were treated as candidate parameter selections, not final model decisions. During the later evaluation phase, some tuned parameter combinations showed signs of overfitting or unstable generalization. Final model selection was based on broader evaluation, not only tuning scores.

---

### 5. Model Evaluation

The model evaluation notebook performed the deepest review of model performance and reliability.

Evaluation techniques included:

- Train/test overfitting checks
- Cross-validation model comparison
- Paired t-test comparison
- Calibration analysis
- ROC-AUC evaluation
- Precision-Recall AUC evaluation
- Threshold tuning
- Error analysis

This phase showed that both Logistic Regression and SVM performed competitively, but Logistic Regression was more stable and interpretable overall.

---

## Final Model Selection

The final selected model was:

```text
Logistic Regression
```

Final model configuration:

```python
LogisticRegression(
    C=1,
    penalty="l1",
    class_weight="balanced",
    solver="liblinear"
)
```

Logistic Regression was selected because it provided:

- Competitive classification performance
- Strong interpretability
- Stable calibration behavior
- Clearer error patterns
- Less sensitivity to noisy engineered features

Although SVM produced competitive results, error analysis showed that it was more sensitive to engineered interaction features and low-information values.

---

## Final Threshold Selection

The default classification threshold of `0.50` was not automatically accepted. Threshold tuning was used to compare multiple probability cutoffs using accuracy, precision, recall, and F1-score.

### Logistic Regression Best Threshold

| Threshold | Accuracy | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|
| 0.45 | 82.46% | 78.07% | 80.18% | 79.11% |

The selected Logistic Regression threshold was:

```text
0.45
```

This threshold produced the strongest balance between precision and recall for the final Logistic Regression model.

### SVM Best Threshold

| Threshold | Accuracy | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|
| 0.40 | 82.84% | 77.31% | 82.88% | 80.00% |

The selected SVM threshold was:

```text
0.40
```

SVM produced a strong threshold-tuned F1-score, but Logistic Regression was still selected as the final model because of its stronger interpretability and more reliable error behavior.

---

## Key Evaluation Results

| Evaluation Area | Logistic Regression | SVM | Interpretation |
|---|---:|---:|---|
| Cross-validation accuracy | 79.0% | 78.2% | Logistic Regression performed slightly better overall |
| Cross-validation recall | 78.6% | 75.1% | Logistic Regression captured more actual survivors across folds |
| Cross-validation F1-score | 74.1% | 72.7% | Logistic Regression had stronger overall balance |
| ROC-AUC | ~88% | ~84% | Logistic Regression showed stronger class separation |
| Best threshold F1-score | 79.11% | 80.00% | SVM peaked slightly higher after threshold tuning |
| Interpretability | Strong | Moderate | Logistic Regression was easier to explain |
| Error behavior | More interpretable | More sensitive to engineered feature noise | Logistic Regression was more stable |

---

## Error Analysis Findings

Error analysis reviewed incorrect predictions by separating false positives and false negatives, then comparing errors across passenger class, sex, age groups, and engineered features.

Key findings:

- Logistic Regression produced more interpretable error patterns.
- Passenger sex remained a strong driver of Logistic Regression prediction behavior.
- SVM errors appeared more connected to engineered feature noise, especially interaction-based variables such as `interaction7` and `Embarked_C`.
- Some engineered features may have introduced unstable signal for SVM rather than improving prediction quality.

This supported the decision to select Logistic Regression as the final model.

---

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- statsmodels
- SciPy
- matplotlib
- seaborn
- ppscore
- XGBoost
- joblib

---

## Skills Demonstrated

This project demonstrates:

- Exploratory data analysis
- Missing-value treatment
- Outlier review and transformation
- Categorical encoding
- Feature engineering
- Predictive feature testing
- Hypothesis testing
- Multicollinearity review
- Classification modeling
- Hyperparameter tuning
- Cross-validation
- Statistical model comparison
- Calibration analysis
- ROC-AUC and PR-AUC evaluation
- Threshold tuning
- Error analysis
- Final model selection
- Model artifact planning

---

## Final Conclusion

This project shows a full machine learning workflow from raw data exploration to final model selection. Logistic Regression was selected as the final model because it balanced predictive performance with interpretability, stability, and reliable error behavior.

The project also showed that strong tuning results do not automatically guarantee the best final model. Broader evaluation, including overfitting checks, calibration, threshold tuning, and error analysis, was necessary to identify the model that generalized best and produced the most understandable prediction behavior.
