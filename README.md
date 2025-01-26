
# Alzheimer Disease Prediction

This project aims to build a machine learning model that predicts whether a person is likely to suffer from Alzheimer's disease based on a set of clinical and demographic features. Alzheimer's disease is a serious medical condition that affects memory and cognitive functions, and early prediction is crucial for timely intervention and treatment.

In this project, we use a dataset containing information about various features, including age, gender, ethnicity, and other medical indicators, to train a machine learning model. The project is implemented in Python using popular data science libraries such as Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, TensorFlow, and XGBoost.


## DATASET
The dataset used in this project consists of multiple features related to patient demographics and clinical measures. It contains:

- **Age:** Age of the patient [years]
- **Gender:** Gender of the patient [Male/Female]
- **Ethnicity:** Ethnicity of the patient [Categorical values]
- **Low_Confidence_Limit:** Lower limit of a statistical confidence interval
- **High_Confidence_Limit:** Upper limit of a statistical confidence interval
- **Geolocation:** Location of the patient [Categorical values]
- **Other Features:** Includes additional relevant indicators
- **Data_Value:** Target variable [1: Alzheimer's disease, 0: Healthy]

The dataset has been preprocessed to handle missing values and encode categorical features for machine-learning purposes.
## Methodology
**Exploratory Data Analysis (EDA):** Insights into the dataset using Pandas, Seaborn, and Matplotlib.
- **Handling Missing Values:** Imputation of missing values using the mean for numerical data and mode for categorical data.
- **Feature Encoding:** Label encoding of categorical variables.

### 2. Data Splitting and Standardization
- The dataset is split into training and testing sets using Scikit-learn's `train_test_split` function.
- Numerical features are standardized using Scikit-learn's `StandardScaler`.

### 3. Machine Learning Models
We trained and evaluated the following models:
- **Neural Network:** A deep learning model implemented using TensorFlow.
- **Random Forest:** An ensemble learning method for classification.
- **XGBoost:** An eXtreme Gradient Boosting model.

### 4. Model Evaluation
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
## Result
After training and evaluating the models, the Neural Network achieved the best performance, with the following results on the test set:
- **Accuracy:** 88.7%
- **Precision:** 86%
- **Recall:** 87%
- **F1 Score:** 88%

## Conclusion
We have successfully built a machine learning model capable of predicting Alzheimer's disease with high accuracy. The project demonstrates the effective use of data preprocessing, feature engineering, and model evaluation.
