import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('india_weather_dataset.csv')

# NORMALIZATION

df.fillna(df.median(numeric_only=True), inplace=True)

df.drop_duplicates(inplace=True)

scaler = StandardScaler()
numeric_cols = ['Latitude', 'Longitude', 'Temperature', 'Humidity', 'Wind Speed']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# datetime
df['Date and Time'] = pd.to_datetime(df['Date and Time'])
df['Year'] = df['Date and Time'].dt.year
df['Month'] = df['Date and Time'].dt.month
df['Day'] = df['Date and Time'].dt.day
df['Hour'] = df['Date and Time'].dt.hour
df.drop(['Date and Time'], axis=1, inplace=True)

#train

df['Rain'] = np.where((df['Humidity'] > 80) & (df['Temperature'] < 25), 1, 0)

X = df[['Latitude', 'Longitude', 'Temperature', 'Humidity', 'Wind Speed']]
y = df['Rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred))


with open('rain_predictor_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('rain_predictor_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("\nEnter the following weather details:")

latitude = float(input("Latitude (e.g., 28.7): "))
longitude = float(input("Longitude (e.g., 77.1): "))
temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
wind_speed = float(input("Wind Speed (km/h): "))

user_input = [[latitude, longitude, temperature, humidity, wind_speed]]
prediction = loaded_model.predict(user_input)[0]

result = "Heavy Rain Expected" if prediction == 1 else "No Heavy Rain Expected"
print(f"\nPrediction: {result}")
