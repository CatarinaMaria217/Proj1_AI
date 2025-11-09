import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Carregar dados
df = pd.read_csv("combined_Lisboa_porto_complete.csv")

# Limpeza igual ao notebook
df.columns = df.columns.str.strip()             # remove espaços
df.columns = df.columns.str.lower()             # minúsculas
df.columns = df.columns.str.replace('.', '_')   # pontos viram _


# Target NO2
y = df['no2']


# Features
numeric_features = [
    'co', 'no2', 'o3', 'pm10', 'pm2_5', 'so2', 
    'temperature_c', 'humidity_percent', 'pressure_hpa', 
    'wind_speed_kmh', 'wind_direction_deg', 'precipitation_mm'
]
categorical_features = ['city']


X = df[numeric_features + categorical_features]


# Pré-processamento
preprocessor = ColumnTransformer([
("num", StandardScaler(), numeric_features),
("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
"Linear Regression": LinearRegression(),
"Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
}


for name, model in models.items():
    reg = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)


print(f"=== {name} ===")
print("MAE:", mean_absolute_error(y_test, preds))
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)