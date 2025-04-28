import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('C:/Users/user/Documents/PythonAI/.venv/data/cars_plus.csv')

X = df[['year', 'engine_volume', 'mileage', 'horsepower', 'brand', 'model']]
y = df['price']

categorical_features = ['brand', 'model']
numerical_features = ['year', 'engine_volume', 'mileage', 'horsepower']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
error_percentage = (mae / y_test.mean()) * 100

print(f"MAE: {mae:.2f} грн")
print(f"R² Score: {r2:.4f}")
print(f"Середня помилка: {error_percentage:.2f}%")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Справжня ціна (грн)')
plt.ylabel('Прогнозована ціна (грн)')
plt.title('Справжня vs Прогнозована ціна (з урахуванням бренду та моделі)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.show()
