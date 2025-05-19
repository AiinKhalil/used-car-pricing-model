

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv("used_cars.csv")
df = df.rename(columns={'brand': 'manufacturer', 'model_year': 'year', 'milage': 'odometer'})
df = df.dropna(subset=['price', 'year', 'odometer', 'manufacturer', 'transmission'])
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['odometer'] = df['odometer'].replace('[\,\smi\.]', '', regex=True).astype(float)
df['year'] = df['year'].astype(int)
df['condition'] = df['accident'].apply(lambda x: 'like new' if 'None reported' in str(x) else 'fair')
df_model = df[['price', 'year', 'odometer', 'manufacturer', 'transmission', 'condition']]

X = df_model[['year', 'odometer', 'manufacturer', 'transmission', 'condition']]
y = df_model['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ['year', 'odometer']
numeric_transformer = StandardScaler()
categorical_features = ['manufacturer', 'transmission', 'condition']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=6, random_state=42))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

# Figure 1
dt_model = dt_pipeline.named_steps['regressor']
ohe = dt_pipeline.named_steps['preprocessor'].named_transformers_['cat']
encoded_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(encoded_feature_names)
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=np.array(feature_names)[indices][:10])
plt.title('Figure 1: Feature Importance - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig("figure1_feature_importance.png")
plt.close()

# Figure 2
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_dt, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.title('Figure 2: Predicted vs. Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.tight_layout()
plt.savefig("figure2_predicted_vs_actual.png")
plt.close()

# Figure 3
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_model, x='condition', y='price')
plt.title('Figure 3: Condition vs. Price')
plt.xlabel('Condition')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.savefig("figure3_condition_vs_price.png")
plt.close()

# Figure 4
numeric_df = df_model[['year', 'odometer', 'price']]
corr_matrix = numeric_df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Figure 4: Correlation Heatmap')
plt.tight_layout()
plt.savefig("figure4_correlation_heatmap.png")
plt.close()
