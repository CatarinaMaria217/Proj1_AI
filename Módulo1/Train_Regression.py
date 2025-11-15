# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error


# # Carregar dados
# df = pd.read_csv("combined_Lisboa_porto_complete.csv")

# # Limpeza igual ao notebook
# df.columns = df.columns.str.strip()             # remove espaços
# df.columns = df.columns.str.lower()             # minúsculas
# df.columns = df.columns.str.replace('.', '_')   # pontos viram _


# # Target NO2
# y = df['no2']


# # Features
# numeric_features = [
#     'co', 'no2', 'o3', 'pm10', 'pm2_5', 'so2', 
#     'temperature_c', 'humidity_percent', 'pressure_hpa', 
#     'wind_speed_kmh', 'wind_direction_deg', 'precipitation_mm'
# ]
# categorical_features = ['city']


# X = df[numeric_features + categorical_features]


# # Pré-processamento
# preprocessor = ColumnTransformer([
# ("num", StandardScaler(), numeric_features),
# ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
# ])


# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# models = {
# "Linear Regression": LinearRegression(),
# "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
# }


# for name, model in models.items():
#     reg = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
#     reg.fit(X_train, y_train)
#     preds = reg.predict(X_test)


# print(f"=== {name} ===")
# print("MAE:", mean_absolute_error(y_test, preds))
# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE:", rmse)

# train_regression.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

def main(args):
    # 1) Load
    data_path = Path(args.data)
    if not data_path.exists():
        alt = Path("combined_Lisboa_porto_complete.csv")
        if alt.exists(): data_path = alt
    df = pd.read_csv(data_path, low_memory=False)

    # 2) Clean headers
    df.rename(columns={c: clean_col(c) for c in df.columns}, inplace=True)

    # 3) Target
    y = df["no2"]  # ajusta se o nome for diferente

    # 4) Features (sem 'no2')
    numeric_features = ["co","o3","pm10","pm2_5","so2",
                        "temperature_c","humidity_percent","pressure_hpa",
                        "wind_speed_kmh","wind_direction_deg","precipitation_mm"]
    categorical_features = ["city"]

    extra_time = []
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["hour"] = df["datetime"].dt.hour
        df["dow"]  = df["datetime"].dt.dayofweek
        extra_time = ["hour","dow"]

    X = df[numeric_features + categorical_features + extra_time]

    # 5) Preprocess (impute + scale + one-hot)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([
        ("num", num_pipe, [c for c in X.columns if c in numeric_features or c in extra_time]),
        ("cat", cat_pipe, [c for c in X.columns if c in categorical_features]),
    ])

    # 6) Split (temporal se possível)
    if "datetime" in df.columns and df["datetime"].notna().any():
        X_train, X_test, y_train, y_test = time_split(X, y, df["datetime"], test_size=args.test_size)
        print("Split usado: temporal")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        print("Split usado: aleatório")

    print("Train/Test:", X_train.shape, X_test.shape)

    # 7) Modelos
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=args.seed),
    }

    # 8) Treino + métricas
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        yh = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, yh)
        rmse = float(np.sqrt(mean_squared_error(y_test, yh)))
        r2   = r2_score(y_test, yh)
        print(f"\n=== {name} ===")
        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/combined_Lisboa_porto_complete.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
