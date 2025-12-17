# P1 – Steel Production Data

## Repository Structure
The repository is organized to clearly separate data, code, and results for the project.

- `/P1-Steel-Production-Data/`
    - `src/`
        - `notebooks/`
            - `1_Preprocessed_csv_data.ipynb`
            - `2_Model_Fitting.ipynb`
            - `3_Model_Comparison.ipynb`
        - `scripts/`
    - `data/`
        - `raw/`
            - `normalized_train_data.xlsx`
            - `normalized_test_data.xlsx`
        - `processed/`
            - `preprocessed_train_data.xlsx`
            - `preprocessed_test_data.xlsx`
    - `results/`
        - `figures/`
            - Model_Comparison_RMSE.png
            - Model_Comparison_MAE.png
            - Model_Comparison_R2.png
        - `tables/`
            - model_comparison.xlsx
            - model_statistics.xlsx
    - `tests/`
    - `README.md`

---

## 1. Topic and Motivation

Steel production is a highly complex industrial process governed by multiple interacting operational, chemical, and environmental parameters. Small variations in these parameters can lead to significant changes in production efficiency, energy consumption, and product quality. Accurate prediction of production-related outcomes is therefore critical for process optimization, cost reduction, and robust decision-making in industrial environments.

Traditional analytical or physics-based models often struggle to represent the inherent non-linearity and uncertainty present in steel production systems. Regression-based machine learning models provide a flexible, data-driven alternative capable of learning complex relationships directly from observed data. This project investigates the applicability and effectiveness of several regression techniques for modeling steel production data.

---

## 2. Related Work

Predictive modeling in industrial manufacturing has been widely studied using statistical and machine learning approaches. Linear regression models are frequently employed due to their simplicity and interpretability, but they are often insufficient for complex industrial processes where non-linear interactions dominate.

Kernel-based methods such as Support Vector Regression (SVR) have demonstrated strong performance in industrial forecasting and process modeling tasks by effectively capturing non-linear dependencies through kernel functions. Ensemble learning methods, including Random Forest regressors, improve robustness by aggregating predictions from multiple decision trees and reducing variance. Neural network–based approaches such as Multi-Layer Perceptrons (MLPs) offer high modeling capacity but typically require large datasets and careful tuning to achieve stable performance.

In this project, representative regression models from linear, kernel-based, ensemble-based, and neural network–based families are implemented and systematically compared on steel production data.

---

## 3. Methods

### 3.1 Data Description and Preprocessing

The dataset used in this study consists of normalized steel production data provided as separate training and testing files. Each observation includes multiple input features representing production-related variables and a continuous target variable representing the production outcome to be predicted.

Raw datasets are stored in the `data/raw/` directory, while cleaned and model-ready datasets are stored in `data/processed/`. Preprocessing steps included verification of data consistency, handling of potential anomalies, and preparation of feature–target mappings. A strict separation between training and test datasets was maintained throughout the workflow to prevent data leakage and ensure reliable evaluation of model generalization performance.

---

### 3.2 Regression Models

The following regression models were implemented and evaluated:

- **Linear Regression**  
  Used as a baseline model to assess linear relationships between input features and the target variable.

- **Support Vector Regression (SVR)**  
  Implemented using a radial basis function (RBF) kernel to model non-linear relationships. Hyperparameters were tuned to balance model flexibility and generalization.

- **Random Forest Regressor**  
  An ensemble-based approach that combines multiple decision trees to model complex feature interactions and improve robustness.

- **Multi-Layer Perceptron (MLP)**  
  A feed-forward neural network trained using backpropagation, included to evaluate the applicability of neural networks for the given dataset.

All models were trained using identical training data and evaluated on the same unseen test dataset to ensure a fair comparison.

---

### 3.3 Evaluation Metrics

Model performance was evaluated using the following standard regression metrics:

- **Root Mean Squared Error (RMSE)**, which penalizes large prediction errors
- **Mean Absolute Error (MAE)**, which measures average absolute deviation
- **Coefficient of Determination (R²)**, which quantifies the proportion of variance explained by the model

These metrics provide complementary insights into accuracy, robustness, and generalization capability.

---

## 4. Results

### 4.1 Evaluation Strategy

All trained models were evaluated on a held-out test dataset that was not used during training. This evaluation strategy ensures that reported results reflect generalization performance rather than training accuracy. Mean values and standard deviations of the evaluation metrics were computed and used to generate comparative bar plots with error bars.

---

### 4.2 Quantitative Results

The quantitative performance of each model on the test dataset is summarized in Table 1.

| Model | RMSE | MAE | R² |
|------|------|------|------|
| Linear Regression | 0.296 | 0.283 | -9.81 |
| Support Vector Regression (SVR) | 0.117 | 0.091 | -0.68 |
| Random Forest Regressor | 0.139 | 0.115 | -1.41 |
| Multi-Layer Perceptron (MLP) | 0.611 | 0.564 | -45.15 |

The mean values and corresponding standard deviations are stored in `results/tables/model_statistics.xlsx`.

---

### 4.3 Error Metric Analysis

The RMSE and MAE plots (available in `results/figures/`) highlight clear performance differences between models. Linear Regression exhibits high error values, confirming its inability to capture non-linear patterns. Random Forest improves upon the linear baseline but remains inferior to SVR.

The MLP model produces the highest error values, indicating unstable learning and poor convergence under the given data conditions. In contrast, Support Vector Regression achieves the lowest RMSE and MAE, demonstrating superior predictive accuracy.

---

### 4.4 R² Analysis

All evaluated models yield negative R² values, indicating that the prediction task is challenging and that the models struggle to fully explain the variance in the target variable. However, Support Vector Regression achieves the least negative R² score, suggesting better explanatory capability compared to the other models.

---

### 4.5 Comparative Discussion

Across all evaluation metrics, **Support Vector Regression consistently outperforms the other models**. Its ability to model smooth non-linear relationships makes it particularly suitable for steel production data. Ensemble and neural network models show limitations under the given dataset size and characteristics, emphasizing the importance of model selection based on data properties.

---

## 5. Limitations

The primary limitation of this study is the limited size of the available dataset, which restricts the performance of data-intensive models such as neural networks. Additionally, the assumption that training and test data follow similar distributions may not fully reflect real-world industrial variability. External factors not captured in the dataset may also influence production outcomes.

---

## 6. Conclusion

This project presented a detailed comparison of regression-based machine learning models applied to steel production data. The results demonstrate that non-linear models significantly outperform linear approaches. **Among all evaluated methods, Support Vector Regression emerged as the best-performing model**, achieving the lowest prediction errors and the most favorable R² score. These findings highlight the importance of selecting appropriate modeling techniques for complex industrial datasets. Future work may explore advanced ensemble methods, feature engineering, and extended cross-validation strategies.

---

## Acknowledgments

- ChatGPT was used for assistance with code structuring, analysis guidance, and documentation drafting
