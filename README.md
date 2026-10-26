# Swiggy Delivery Time Prediction using Machine Learning

## Project Overview

This project aims to predict food delivery time in minutes using machine learning regression techniques. It uses historical delivery data containing information about delivery partners, weather conditions, traffic, order details, vehicle conditions, and other factors that may influence delivery time.

The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, feature selection, model comparison, hyperparameter optimization, and deployment using Streamlit.

The final model uses an optimized XGBoost Regressor to predict the expected delivery time based on user-provided order details.

## Business Problem

Accurate delivery-time estimation is important for food delivery platforms. Inaccurate predictions can lead to customer dissatisfaction, delivery delays, and inefficient delivery planning.

The objective of this project is to build a machine learning model that predicts delivery time more accurately using historical order and delivery information.

## Project Objectives

- Analyze historical food delivery data.
- Perform exploratory data analysis to understand patterns and relationships.
- Handle missing values and prepare numerical and categorical features.
- Engineer useful features, including distance and time-related variables.
- Compare multiple regression algorithms.
- Select important features using SelectKBest.
- Optimize XGBoost hyperparameters using Optuna.
- Evaluate model performance using RMSE and R².
- Deploy the trained model using Streamlit.

## Dataset Description

The dataset contains information related to delivery partners, orders, and delivery conditions.

Important features include:

| Feature | Description |
|---|---|
| age | Age of the delivery partner |
| ratings | Delivery partner rating |
| weather | Weather conditions |
| traffic | Traffic intensity |
| vehicle_condition | Condition of the delivery vehicle |
| type_of_order | Type of food order |
| type_of_vehicle | Vehicle used for delivery |
| multiple_deliveries | Number of deliveries handled together |
| festival | Festival indicator |
| city_type | Type of city |
| city_name | City identifier |
| order_day | Day of the month |
| order_month | Month of the order |
| order_day_of_week | Day of the week |
| is_weekend | Whether the order was placed on a weekend |
| pickup_time_minutes | Time taken for pickup |
| order_time_hour | Hour of the order |
| order_time_of_day | Time-of-day category |
| distance | Distance between restaurant and customer |
| time_taken | Target variable: delivery time in minutes |

The latitude and longitude columns were excluded from the final deployment inputs, while the derived distance feature was retained.

**Target variable:** `time_taken`

**Problem type:** Supervised Machine Learning — Regression

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Optuna
- Joblib
- Streamlit
- Jupyter Notebook

## Project Workflow

### 1. Exploratory Data Analysis

- Examined dataset structure and data types.
- Analyzed missing values and duplicate records.
- Studied numerical feature distributions and skewness.
- Investigated outliers and relationships between variables.
- Examined factors that may influence delivery time.

### 2. Data Preprocessing

Different preprocessing strategies were applied according to feature type.

- Mean imputation for selected numerical features.
- Median imputation for selected numerical features.
- Most-frequent imputation for categorical features.
- One-Hot Encoding for nominal categorical variables.
- Ordinal Encoding for traffic intensity.
- RobustScaler for numerical feature scaling.
- ColumnTransformer and pipelines for organized preprocessing.

### 3. Feature Engineering

Created useful features such as distance, order date components, weekend indicators, order hour, time-of-day categories, and pickup time.

These features help the model learn relationships between order characteristics and delivery duration.

### 4. Feature Selection

SelectKBest with `f_regression` was used to rank features according to their individual statistical relationship with the continuous target variable.

The top 30 features were selected for model training.

Feature selection was fitted on the training data and then applied to the test data to avoid data leakage.

### 5. Model Training and Comparison

Several regression algorithms were evaluated, including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regression
- XGBoost Regressor

XGBoost provided the strongest performance among the initial models evaluated.

### 6. Hyperparameter Optimization

Optuna was used to optimize the XGBoost model.

Important hyperparameters included:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

The optimization process searched for a parameter combination that improved predictive performance.

### 7. Model Evaluation

The final model was evaluated using:

**Root Mean Squared Error (RMSE):** Measures prediction error in the same unit as delivery time, minutes. Lower values are better.

**R² Score:** Measures how much variation in the target is explained by the model. Higher values generally indicate better predictive performance.

## Results

The optimized XGBoost model achieved the following approximate test-set results:

| Metric | Result |
|---|---:|
| RMSE | 3.92 minutes |
| R² Score | 0.825 |

An RMSE of approximately 3.92 minutes indicates the scale of prediction error, with larger errors receiving greater weight. It should not be interpreted as the exact average absolute error.

An R² score of approximately 0.825 indicates that the model explains around 82.5% of the variation in delivery time on the evaluated test set.

Results may vary depending on the dataset version, preprocessing, and train-test split.

## Deployment Using Streamlit

The trained model was deployed through a Streamlit web application.

Users can enter delivery-related information and obtain a predicted delivery time in minutes.

The application uses three saved components:

- `preprocessor.pkl` — transforms input data.
- `feature_selector.pkl` — selects the trained features.
- `xgb_model.pkl` — generates the delivery-time prediction.

### Run the Application Locally

**1. Clone the repository**

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start the Streamlit application**

```bash
python -m streamlit run app.py
```

**4. Open the local URL**

Streamlit will display a local address, usually:

```text
http://localhost:8501
```

Open this address in your browser.

## Project Structure

```text
Swiggy-Delivery-Time-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── preprocessor.pkl
├── feature_selector.pkl
├── xgb_model.pkl
└── notebooks/
    └── swiggy_delivery_time_prediction.ipynb
```

The notebook folder is optional; adjust this structure to match the files actually included in your repository.

## Key Learnings

- Applying data cleaning and exploratory analysis to a real-world regression problem.
- Building reusable preprocessing pipelines with Scikit-learn.
- Handling missing values, categorical encoding, and numerical scaling.
- Selecting features using statistical techniques.
- Comparing regression algorithms using appropriate evaluation metrics.
- Optimizing XGBoost hyperparameters with Optuna.
- Saving trained components with Joblib.
- Integrating a machine learning model into a Streamlit application.

## Future Improvements

- Incorporate live traffic and weather information.
- Evaluate model performance on additional unseen data.
- Monitor prediction errors after deployment.
- Retrain the model periodically as delivery patterns change.
- Deploy the application to a cloud hosting platform.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for food delivery-time prediction, from data understanding and preprocessing to model optimization and interactive deployment.

The optimized XGBoost Regressor achieved an approximate RMSE of 3.92 minutes and an R² score of 0.825 on the evaluated test set. The Streamlit application provides a practical interface for generating delivery-time predictions.

## Author

**Meghna Palwai**

Machine Learning | Data Science | Python
