import pandas as pd
import matplotlib.pyplot as plt

data = {
    'OrderID': [1, 2, 3, 4, 5, 6, 7],
    'CustomerName': ['Ivanov', 'Petrenko', 'Shevchenko', 'Ivanov', 'Bondarenko', 'Petrenko', 'Shevchenko'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Furniture', 'Furniture', 'Electronics'],
    'Product': ['Laptop', 'T-shirt', 'Smartphone', 'Jeans', 'Table', 'Chair', 'Tablet'],
    'Quantity': [1, 2, 1, 3, 1, 4, 2],
    'Price': [30000, 500, 20000, 700, 5000, 1500, 15000],
    'OrderDate': ['2024-06-04', '2024-06-05', '2024-06-06', '2024-06-09', '2024-06-10', '2024-06-11', '2024-06-12']
}

df = pd.DataFrame(data)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

df['TotalAmount'] = df['Quantity'] * df['Price']

print('Сумарний дохід магазину:', df['TotalAmount'].sum())

print('Середнє TotalAmount:', df['TotalAmount'].mean())

print('Кількість замовлень по кожному клієнту:')
print(df['CustomerName'].value_counts())

print('Замовлення, де сума покупки > 500:')
print(df[df['TotalAmount'] > 500])

print('Таблиця за датою (зворотній порядок):')
print(df.sort_values('OrderDate', ascending=False))

print('Замовлення з 5 по 10 червня:')
mask = (df['OrderDate'] >= '2024-06-05') & (df['OrderDate'] <= '2024-06-10')
print(df[mask])

grouped = df.groupby('Category').agg(
    Products=('Quantity', 'sum'),
    TotalSales=('TotalAmount', 'sum')
)
print('Групування за категорією:')
print(grouped)

top_customers = df.groupby('CustomerName')['TotalAmount'].sum().sort_values(ascending=False).head(3)
print('ТОП-3 клієнтів за сумою покупок:')
print(top_customers)

orders_per_date = df.groupby('OrderDate').size()
orders_per_date.plot(kind='line', marker='o')
plt.title('Кількість замовлень по датах')
plt.xlabel('Дата')
plt.ylabel('Кількість замовлень')
plt.grid(True)
plt.show()

income_per_category = df.groupby('Category')['TotalAmount'].sum()
income_per_category.plot(kind='bar')
plt.title('Доходи по категоріях')
plt.xlabel('Категорія')
plt.ylabel('Доходи')
plt.grid(True)
plt.show()
