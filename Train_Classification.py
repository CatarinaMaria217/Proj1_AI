

# train_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar dados
df = pd.read_csv("combined_Lisboa_porto_complete.csv")
df.columns = df.columns.str.strip()  # remover espaços extras

# Limpeza igual ao notebook
df.columns = df.columns.str.strip()             # remove espaços
df.columns = df.columns.str.lower()             # minúsculas
df.columns = df.columns.str.replace('.', '_')   # pontos viram _

# Converter target para 0/1
df["label_air_good"] = df["label_air_good"].map({"Yes": 1, "No": 0})

# Definir features
numeric_features = [
    'co', 'no2', 'o3', 'pm10', 'pm2_5', 'so2', 
    'temperature_c', 'humidity_percent', 'pressure_hpa', 
    'wind_speed_kmh', 'wind_direction_deg', 'precipitation_mm'
]
categorical_features = ['city']

# Pré-processamento
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Variáveis
X = df[numeric_features + categorical_features]
y = df['label_air_good']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Modelos
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=17)
}

# Treinamento e avaliação
for name, model in models.items():
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=['Bad Air', 'Good Air']))