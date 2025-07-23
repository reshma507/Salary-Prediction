# Employee Salary Prediction using Machine Learning

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Dataset
df = pd.read_csv('Salary_Data.csv')
df.dropna(inplace=True)

# Reduce Job Titles (less than 25 counts -> "Others")
job_counts = df['Job Title'].value_counts()
rare_jobs = job_counts[job_counts <= 25].index
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in rare_jobs else x)

# Clean Education Levels
df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
}, inplace=True)

# Encode Education Level numerically
education_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_map)

# Label Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-Hot Encode Job Title
job_dummies = pd.get_dummies(df['Job Title'], drop_first=True)
df = pd.concat([df.drop('Job Title', axis=1), job_dummies], axis=1)

# Features and Target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split into Train/Test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define Models and Hyperparameters
model_params = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [2, 4, 6, 8, 10],
            'random_state': [0, 42],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 20, 30, 50]
        }
    }
}

# Grid Search CV for Model Selection
scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_mean_squared_error')
    clf.fit(x_train, y_train)
    scores.append({
        'Model': model_name,
        'Best Params': clf.best_params_,
        'MSE': -clf.best_score_
    })

score_df = pd.DataFrame(scores)
print("GridSearchCV Results:\n", score_df.sort_values(by='MSE'))

# Train Best Models and Evaluate
# 1. Random Forest
rf = RandomForestRegressor(n_estimators=20)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_r2 = rf.score(x_test, y_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)

# 2. Decision Tree
dt = DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=0)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
dt_r2 = dt.score(x_test, y_test)
dt_mae = mean_absolute_error(y_test, y_pred_dt)
dt_rmse = mean_squared_error(y_test, y_pred_dt, squared=False)

# 3. Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
lr_r2 = lr.score(x_test, y_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = mean_squared_error(y_test, y_pred_lr, squared=False)

# Print All Metrics
print("\n--- Model Performance ---")
print("Random Forest → R²:", rf_r2, "MAE:", rf_mae, "RMSE:", rf_rmse)
print("Decision Tree → R²:", dt_r2, "MAE:", dt_mae, "RMSE:", dt_rmse)
print("Linear Regression → R²:", lr_r2, "MAE:", lr_mae, "RMSE:", lr_rmse)

# Bar Chart Comparing R² Scores
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r2_scores = [lr_r2, dt_r2, rf_r2]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=r2_scores, palette='Set2')
plt.title('Model Comparison - R² Score')
plt.ylabel('R² Score')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.show()
