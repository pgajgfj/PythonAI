
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

def is_night(hour):
    return int(hour >= 22 or hour < 6)

def prepare_data():
    X, y = [], []
    for hour in range(24):
        for minute in [0, 30]:
            total_minutes = hour * 60 + minute
            night = is_night(hour)
            X.append([total_minutes, night])
            duration = 20 + (5 * night) + np.random.normal(0, 2)
            y.append(duration)
    return np.array(X), np.array(y)


X, y = prepare_data()


model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=500, random_state=42)
model.fit(X, y)


joblib.dump(model, "mlp_model.pkl")


def predict_trip_duration(hour, minute):
    total_minutes = hour * 60 + minute
    night = is_night(hour)
    return model.predict([[total_minutes, night]])[0]


print("🕙 10:30 →", round(predict_trip_duration(10, 30), 2), "хв")
print("🌙 00:00 →", round(predict_trip_duration(0, 0), 2), "хв")
print("🌙 02:40 →", round(predict_trip_duration(2, 40), 2), "хв")
