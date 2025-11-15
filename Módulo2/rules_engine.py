
import json, pandas as pd

def evaluate_rule_text(rule_text, alert):
    expr = rule_text.lower().replace("se ", "").replace("então ", "->")
    expr = expr.replace(" e ", " and ").replace(" ou ", " or ")
    expr = expr.replace("temp", "alert['temp']").replace("hum", "alert['hum']").replace("wind", "alert['wind']")
    expr = expr.replace("event_type", "alert['event_type']")
    cond, action = expr.split("->")
    cond = cond.strip()
    action = action.strip()
    try:
        if eval(cond):
            return action
    except Exception:
        return None
    return None

def run_engine(alerts_df, rules_list):
    recs = []
    for _, row in alerts_df.iterrows():
        alert = row.to_dict()
        alert['event_type'] = str(alert.get('event_type')).lower()
        actions = []
        for r in rules_list:
            act = evaluate_rule_text(r['rule'], alert)
            if act:
                actions.append({"rule_id": r["id"], "action": act})
        if not actions:
            actions = [{"rule_id": None, "action": "monitor"}]
        recs.append({"timestamp": alert.get('timestamp'), "zone": alert.get('zone'), "actions": actions})
    out = pd.DataFrame([
        {"timestamp": r["timestamp"], "zone": r["zone"], "actions": ";".join([a["action"] for a in r["actions"]])}
        for r in recs
    ])
    return out

if __name__ == "__main__":
    import sys, json
    alerts = pd.read_csv("alerts.csv")
    rules = json.load(open("regras.json", encoding="utf-8"))
    out = run_engine(alerts, rules)
    out.to_csv("recommendations.csv", index=False)


"""1. SE temp > 40 E hum < 20 → risco_incendio_alto

Quando a temperatura ultrapassa os 40 °C e a humidade é muito baixa, o ambiente fica extremamente propenso a ignições rápidas.
➡️ Conclusão: As condições são ideais para iniciar e propagar incêndios florestais ou urbanos.

2. SE temp > 35 E wind > 40 → risco_incendio_moderado

Temperaturas elevadas combinadas com ventos fortes favorecem a propagação de chamas.
➡️ Conclusão: Há risco relevante de incêndio, embora menos extremo do que na regra 1.

3. SE event_type = 'fumo' E hum < 30 → risco_incendio_alto

A presença de fumo já é um indício direto de combustão. Com humidade baixa, o risco aumenta drasticamente.
➡️ Conclusão: Pode existir um incêndio ativo ou a iniciar-se.

4. SE event_type = 'ruido' E wind > 50 → risco_acidente_via

Ruído intenso pode indicar queda de objetos, colisões ou ruturas. Com ventos fortes, aumenta o risco de árvores tombarem ou objetos serem projetados.
➡️ Conclusão: Pode ter ocorrido um acidente (por exemplo, queda de estruturas ou acidentes rodoviários).

5. SE temp > 40 E hum < 10 E wind > 30 → risco_incendio_critico

Combinação extrema: calor muito alto, humidade quase inexistente e ventos moderados/fortes.
➡️ Conclusão: Situação de máxima severidade, com probabilidade muito alta de incêndio explosivo.

6. SE temp ≥ 30 E hum < 20 → vigilancia_fortemente_recomendada

Mesmo abaixo de condições críticas, já existe risco acrescido de ignição ou desidratação ambiental.
➡️ Conclusão: As autoridades devem reforçar vigilância e monitorização ativa.

7. SE event_type = 'fumo' E wind < 20 → possivel_fuga_quimica

Se há detecção de fumo mas o vento é baixo, o fumo não se dispersa — pode indicar:

fumaça densa e localizada,

combustão incompleta,

fuga química ou industrial.
➡️ Conclusão: Investigar possível incidente industrial ou químico.

8. SE hum < 15 E temp > 38 → reduzir_atividades_ao_ar_livre

Humidade muito baixa e calor intenso aumentam risco para pessoas (stress térmico) e para início de incêndios.
➡️ Conclusção: Recomenda-se limitar atividades outdoor, sobretudo esforço físico."""