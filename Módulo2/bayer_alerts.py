
# Simple Bayesian network for Calor, Humidade, RiscoIncendio
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
    print(enumeration_ask({"Calor":"High","Humidade":"Low"}))
