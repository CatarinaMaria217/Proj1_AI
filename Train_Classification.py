

# # train_classification.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Carregar dados
# df = pd.read_csv("combined_Lisboa_porto_complete.csv")
# df.columns = df.columns.str.strip()  # remover espaços extras

# # Limpeza igual ao notebook
# df.columns = df.columns.str.strip()             # remove espaços
# df.columns = df.columns.str.lower()             # minúsculas
# df.columns = df.columns.str.replace('.', '_')   # pontos viram _

# # Converter target para 0/1
# df["label_air_good"] = df["label_air_good"].map({"Yes": 1, "No": 0})

# # Definir features
# numeric_features = [
#     'co', 'no2', 'o3', 'pm10', 'pm2_5', 'so2', 
#     'temperature_c', 'humidity_percent', 'pressure_hpa', 
#     'wind_speed_kmh', 'wind_direction_deg', 'precipitation_mm'
# ]
# categorical_features = ['city']

# # Pré-processamento
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numeric_features),
#     ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
# ])

# # Variáveis
# X = df[numeric_features + categorical_features]
# y = df['label_air_good']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Modelos
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000),
#     "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
# }

# # Treinamento e avaliação
# for name, model in models.items():
#     clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)
#     print(f"\n=== {name} ===")
#     print("Accuracy:", accuracy_score(y_test, preds))
#     print(classification_report(y_test, preds, target_names=['Bad Air', 'Good Air']))

# train_classification.py
import argparse
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Utils
# ---------------------------
def clean_col(c: str) -> str:
    c = c.strip()
    if c.startswith("#"): c = c[1:].strip()
    return (c.lower()
             .replace(" ", "_")
             .replace(".", "_")
             .replace("%", "percent")
             .replace("º","").replace("°",""))

def time_split(X: pd.DataFrame, y: pd.Series, dt: pd.Series, test_size=0.2):
    order = pd.to_datetime(dt, errors="coerce").sort_values().index
    cut = int((1 - test_size) * len(order))
    tr_idx, te_idx = order[:cut], order[cut:]
    return X.loc[tr_idx], X.loc[te_idx], y.loc[tr_idx], y.loc[te_idx]

# ---------------------------
# Main
# ---------------------------
def main(args):
    # 1) Carregar dados
    data_path = Path(args.data)
    if not data_path.exists():
        # fallback comum
        alt = Path("combined_Lisboa_porto_complete.csv")
        if alt.exists(): data_path = alt
    df = pd.read_csv(data_path, low_memory=False)

    # 2) Limpar cabeçalhos
    df.rename(columns={c: clean_col(c) for c in df.columns}, inplace=True)

    # 3) Target 0/1
    if df["label_air_good"].dtype == object:
        df["label_air_good"] = df["label_air_good"].map({"Yes": 1, "No": 0})

    # 4) Features
    numeric_features = [
        "co","no2","o3","pm10","pm2_5","so2",
        "temperature_c","humidity_percent","pressure_hpa",
        "wind_speed_kmh","wind_direction_deg","precipitation_mm"
    ]
    categorical_features = ["city"]

    # (opcional) features temporais auxiliares
    extra_time = []
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["hour"] = df["datetime"].dt.hour
        df["dow"]  = df["datetime"].dt.dayofweek
        extra_time = ["hour","dow"]

    X = df[numeric_features + categorical_features + extra_time]
    y = df["label_air_good"]

    # 5) Pré-processamento (imputação + escala + one-hot)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, [c for c in X.columns if c in numeric_features or c in extra_time]),
        ("cat", cat_pipe, [c for c in X.columns if c in categorical_features]),
    ])

    # 6) Split: temporal se houver datetime válido; senão, estratificado
    if "datetime" in df.columns and df["datetime"].notna().any():
        X_train, X_test, y_train, y_test = time_split(X, y, df["datetime"], test_size=args.test_size)
        split_used = "temporal"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        split_used = "estratificado"

    print(f"Split usado: {split_used}")
    print("Train/Test:", X_train.shape, X_test.shape)

    # 7) Modelos
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=args.seed, class_weight="balanced_subsample"
        ),
    }

    # 8) Treino + avaliação
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        print(f"\n=== {name} ===")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds, target_names=["Bad Air","Good Air"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/combined_Lisboa_porto_complete.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
