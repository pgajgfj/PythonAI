

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)
y = x**2

plt.figure(figsize=(6, 4))
plt.plot(x, y)
plt.title('Графік функції y = x²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


mu, sigma = 5, 2
data = np.random.normal(mu, sigma, 1000)

plt.figure(figsize=(6, 4))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Гістограма нормального розподілу')
plt.xlabel('Значення')
plt.ylabel('Частота')
plt.grid(True)
plt.show()

hobbies = ['Читання', 'Спорт', 'Малювання', 'Подорожі', 'Ігри']
time_spent = [20, 30, 15, 25, 10]

plt.figure(figsize=(6, 6))
plt.pie(time_spent, labels=hobbies, autopct='%1.1f%%', startangle=140)
plt.title('Мої хобі')
plt.axis('equal')
plt.show()

fruits = ['Яблука', 'Банани', 'Апельсини', 'Груші']
data_fruits = [
    np.random.normal(150, 20, 100),
    np.random.normal(120, 15, 100),
    np.random.normal(180, 25, 100),
    np.random.normal(130, 10, 100)
]

plt.figure(figsize=(6, 4))
plt.boxplot(data_fruits, labels=fruits)
plt.title('Box-plot для маси фруктів')
plt.ylabel('Маса (грам)')
plt.grid(True)
plt.show()


x_points = np.random.uniform(0, 1, 100)
y_points = np.random.uniform(0, 1, 100)

plt.figure(figsize=(6, 4))
plt.scatter(x_points, y_points, color='green', alpha=0.6)
plt.title('Точкова діаграма')
plt.xlabel('Вісь X')
plt.ylabel('Вісь Y')
plt.grid(True)
plt.show()


x = np.linspace(-5, 5, 400)
y1 = x
y2 = x**2
y3 = x**3

plt.figure(figsize=(6, 4))
plt.plot(x, y1, label='y = x', color='red')
plt.plot(x, y2, label='y = x²', color='blue')
plt.plot(x, y3, label='y = x³', color='green')

plt.title('Графіки функцій')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
