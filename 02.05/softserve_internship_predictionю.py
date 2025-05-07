import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('C:/Users/user/Documents/PythonAI/.venv/data/internship_candidates_cefr_final.csv')# шлях вказаний так тому що, по іншому не працює, чому так я хз

le = LabelEncoder()
df['EnglishLevelEncoded'] = le.fit_transform(df['EnglishLevel'])

X = df[['Experience', 'Grade', 'EnglishLevelEncoded', 'Age', 'EntryTestScore']]
y = df['Accepted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

english_range = np.arange(0, len(le.classes_))
entry_range = np.linspace(df['EntryTestScore'].min(), df['EntryTestScore'].max(), 100)

xx, yy = np.meshgrid(english_range, entry_range)
Z = model.predict_proba(np.c_[np.full(xx.ravel().shape, df['Experience'].median()),
                               np.full(xx.ravel().shape, df['Grade'].median()),
                               xx.ravel(),
                               np.full(xx.ravel().shape, df['Age'].median()),
                               yy.ravel()])[:,1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
cp = plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm')
plt.colorbar(cp, label='Ймовірність прийняття')
plt.xlabel('Рівень англійської (кодування)')
plt.ylabel('Бали за вступний тест')
plt.title('Ймовірність прийняття залежно від EnglishLevel та EntryTestScore')
plt.xticks(english_range, le.classes_, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
