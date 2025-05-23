import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
days = np.arange(1, 366)
energy_consumption = 100 + 30 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 5, size=365)

df = pd.DataFrame({'day_of_year': days, 'consumption': energy_consumption})

scaler = MinMaxScaler()
X = scaler.fit_transform(df[['day_of_year']])
y = df['consumption'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    full_preds = model(torch.tensor(X, dtype=torch.float32)).numpy()

print(f'{test_loss.item():.2f}')

plt.figure(figsize=(10, 5))
plt.plot(df['day_of_year'], df['consumption'], label='Фактичне')
plt.plot(df['day_of_year'], full_preds.flatten(), label='Прогнозоване')
plt.xlabel('День року')
plt.ylabel('Споживання енергії')
plt.legend()
plt.title('Прогноз споживання електроенергії')
plt.show()
