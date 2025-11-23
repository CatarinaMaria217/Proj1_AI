
"""# Simple Bayesian network for Calor, Humidade, RiscoIncendio
P_Calor = {"High": 0.1, "Normal": 0.9}
P_Humidade = {"Low": 0.2, "Normal": 0.8}
P_Risco = {
    ("High", "Low"): {"High": 0.95, "Low": 0.05},
    ("High", "Normal"): {"High": 0.7, "Low": 0.3},
    ("Normal", "Low"): {"High": 0.6, "Low": 0.4},
    ("Normal", "Normal"): {"High": 0.01, "Low": 0.99},
}

def enumeration_ask(obs):
    Q = {"High":0.0, "Low":0.0}
    for risco_state in ["High","Low"]:
        total = 0.0
        for calor_state in ["High","Normal"]:
            for hum_state in ["Low","Normal"]:
                prob = P_Calor[calor_state] * P_Humidade[hum_state] * P_Risco[(calor_state, hum_state)][risco_state]
                # apply observations
                if "Calor" in obs and obs["Calor"] != calor_state:
                    prob = 0.0
                if "Humidade" in obs and obs["Humidade"] != hum_state:
                    prob = 0.0
                total += prob
        Q[risco_state] = total
    s = Q["High"] + Q["Low"]
    Q["High"] /= s
    Q["Low"] /= s
    return Q

if __name__ == "__main__":
    print(enumeration_ask({"Calor":"High","Humidade":"Low"}))"""

import pandas as pd

# =============================================================
# 1. LER O DATASET
# =============================================================
df = pd.read_csv("alerts.csv")   # <-- caminho do ficheiro enviado

# =============================================================
# 2. DEFINIR LIMIARES AUTOMÁTICOS A PARTIR DO DATASET
# =============================================================
# Quartis: isto adapta os thresholds aos dados reais
temp_high_threshold = df["temp"].quantile(0.75)   # acima disto é High
hum_low_threshold = df["hum"].quantile(0.25)      # abaixo disto é Low

print("Limiar Calor High:", temp_high_threshold)
print("Limiar Humidade Low:", hum_low_threshold)

# =============================================================
# 3. FUNÇÕES DE CATEGORIZAÇÃO (CONVERTE NÚMEROS EM High/Normal)
# =============================================================
def categorize_calor(temp):
    return "High" if temp > temp_high_threshold else "Normal"

def categorize_humidade(hum):
    return "Low" if hum < hum_low_threshold else "Normal"

# =============================================================
# 4. MODELO BAYESIANO
# =============================================================
P_Calor = {"High": 0.1, "Normal": 0.9}
P_Humidade = {"Low": 0.2, "Normal": 0.8}

P_Risco = {
    ("High", "Low"): {"High": 0.95, "Low": 0.05},
    ("High", "Normal"): {"High": 0.7, "Low": 0.3},
    ("Normal", "Low"): {"High": 0.6, "Low": 0.4},
    ("Normal", "Normal"): {"High": 0.01, "Low": 0.99},
}

def enumeration_ask(obs):
    Q = {"High": 0.0, "Low": 0.0}

    for risco_state in ["High", "Low"]:
        total = 0.0
        for calor_state in ["High", "Normal"]:
            for hum_state in ["Low", "Normal"]:
                prob = (
                    P_Calor[calor_state]
                    * P_Humidade[hum_state]
                    * P_Risco[(calor_state, hum_state)][risco_state]
                )

                # Observações aplicadas
                if "Calor" in obs and obs["Calor"] != calor_state:
                    prob = 0.0
                if "Humidade" in obs and obs["Humidade"] != hum_state:
                    prob = 0.0

                total += prob

        Q[risco_state] = total

    s = Q["High"] + Q["Low"]
    Q["High"] /= s
    Q["Low"] /= s
    return Q

# =============================================================
# 5. APLICAR O MODELO AO DATASET COMPLETO
# =============================================================
risco_list = []

for _, row in df.iterrows():
    calor_cat = categorize_calor(row["temp"])
    hum_cat = categorize_humidade(row["hum"])

    resultado = enumeration_ask({"Calor": calor_cat, "Humidade": hum_cat})

    risco_list.append({
        "timestamp": row["timestamp"],
        "zone": row["zone"],
        "temp": row["temp"],
        "hum": row["hum"],
        "wind": row["wind"],
        "Calor_cat": calor_cat,
        "Humidade_cat": hum_cat,
        "Prob_Incêndio_Alto": resultado["High"],
        "Prob_Incêndio_Baixo": resultado["Low"],
    })

# Converter em DataFrame
df_risco = pd.DataFrame(risco_list)

# =============================================================
# 6. GUARDAR RESULTADO
# =============================================================
df_risco.to_csv("risco_saida.csv", index=False)
print("Ficheiro gerado: risco_saida.csv")
