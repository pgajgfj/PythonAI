import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

X = np.linspace(-20, 20, 400).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * X.ravel()**2

plt.figure(figsize=(6, 4))
plt.plot(X, y, color='blue')
plt.title('Графік функції f(x) = sin(x) + 0.1x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train, label='Тренувальні дані', color='skyblue', alpha=0.6)
plt.scatter(X_test, y_test, label='Тестові дані', color='orange', alpha=0.6)
plt.title('Розподіл тренувальних і тестових даних')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                     solver='adam', max_iter=5000, random_state=42)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_all = model.predict(X)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

print(f'MAE (Середня абсолютна помилка): {mae:.3f}')
print(f'MSE (Середньоквадратична помилка): {mse:.3f}')

plt.figure(figsize=(6, 4))
plt.plot(X, y, label='Реальна функція', color='blue')
plt.plot(X, y_pred_all, label='Передбачення моделі', color='red', linestyle='--')
plt.title('Реальна та передбачувана функція')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

errors = y_test - y_pred_test

plt.figure(figsize=(6, 4))
plt.hist(errors, bins=30, color='lightgreen', edgecolor='black')
plt.title('Гістограма помилок передбачення')
plt.xlabel('Помилка (y_test - y_pred)')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_test, color='purple', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Ідеал')
plt.title('Передбачене vs Реальне')
plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.legend()
plt.grid(True)
plt.show()
