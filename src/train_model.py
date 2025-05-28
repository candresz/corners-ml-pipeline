#!/usr/bin/env python
# coding: utf-8

# In[1]:


# === IMPORTS ===
# === IMPORTS ===
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
import re
import warnings
import shap
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# === CONFIGURACIÓN API ===
# === API CONFIGURATION ===
API_KEY = "API KEY"
headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# === DEFINICIONES ===
# === FUNCTION DEFINITIONS ===
features_interes = [
    "Ball Possession", "Total Shots", "Blocked Shots", "Shots on Target", "Shots off Target",
    "Fouls", "Yellow Cards", "Red Cards", "Dangerous Attacks", "Free Kicks"
]

def to_numeric_safe(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '').str.replace('None', '').str.replace('null', ''), errors='coerce')
    return df

def asignar_binario(corners):
    return 1 if corners >= 12 else 0

def weighted_average(series, weights):
    if series.empty:
        return 0
    return np.average(series, weights=weights[:len(series)])

def es_equipo_under(stats, umbral=0.8):
    under_count = 0
    total = 0
    for match in stats:
        corners = match.get("home_Corner Kicks") or match.get("away_Corner Kicks")
        if pd.notna(corners):
            total += 1
            if corners <= 11:
                under_count += 1
    return total >= 3 and (under_count / total) >= umbral

def agregar_features_tacticos(df):
    # Disparos por posesión
    if "home_Total Shots" in df.columns and "home_Ball Possession" in df.columns:
        df["shots_per_possession_home"] = df["home_Total Shots"] / (df["home_Ball Possession"] + 1)
    if "away_Total Shots" in df.columns and "away_Ball Possession" in df.columns:
        df["shots_per_possession_away"] = df["away_Total Shots"] / (df["away_Ball Possession"] + 1)

    # Corner por ataque peligroso
    if "home_Corner Kicks" in df.columns and "home_Dangerous Attacks" in df.columns:
        df["corner_efficiency_home"] = df["home_Corner Kicks"] / (df["home_Dangerous Attacks"] + 1)
    if "away_Corner Kicks" in df.columns and "away_Dangerous Attacks" in df.columns:
        df["corner_efficiency_away"] = df["away_Corner Kicks"] / (df["away_Dangerous Attacks"] + 1)

    # Eficiencia de disparos
    if "home_Shots on Target" in df.columns and "home_Total Shots" in df.columns:
        df["shots_efficiency_home"] = df["home_Shots on Target"] / (df["home_Total Shots"] + 1)
    if "away_Shots on Target" in df.columns and "away_Total Shots" in df.columns:
        df["shots_efficiency_away"] = df["away_Shots on Target"] / (df["away_Total Shots"] + 1)

    return df

# === ENTRENAMIENTO ===
# === TRAINING ===

matches = []
for season in range(2016, 2026):
    params = {"league": 239, "season": season}
    response = requests.get("https://api-football-v1.p.rapidapi.com/v3/fixtures", headers=headers, params=params)
    if response.status_code == 200:
        matches += response.json()["response"]

fixtures = pd.json_normalize(matches)
if "fixture.status.short" not in fixtures.columns:
    fixtures["fixture.status.short"] = None

ultimos_completados = fixtures[fixtures["fixture.status.short"] == "FT"].tail(4000)
if ultimos_completados.empty:
    raise ValueError("❌ No hay matches finalizados disponibles para entrenar el modelo.")

historial = []
for _, row in tqdm(ultimos_completados.iterrows(), total=ultimos_completados.shape[0]):
    fixture_id = row["fixture.id"]
    home_team = row["teams.home.name"]
    away_team = row["teams.away.name"]
    stats_url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    res = requests.get(stats_url, headers=headers, params={"fixture": fixture_id})
    if res.status_code == 200:
        stats = res.json()["response"]
        partido = {}
        for stat in stats:
            side = "home" if stat["team"]["name"] == home_team else "away"
            for item in stat["statistics"]:
                if item["type"] in features_interes or item["type"] == "Corner Kicks":
                    partido[f"{side}_{item['type']}"] = item["value"]
        if partido:
            partido["home_team"] = home_team
            partido["away_team"] = away_team
            historial.append(partido)

# === PROCESAMIENTO ===
# === DATA PROCESSING ===
df = pd.DataFrame(historial)
df = to_numeric_safe(df, df.columns)
df = agregar_features_tacticos(df)


df["corners_total"] = df["home_Corner Kicks"] + df["away_Corner Kicks"]
df["corners_binario"] = df["corners_total"].apply(asignar_binario)
df["possession_diff"] = df["home_Ball Possession"] - df["away_Ball Possession"]
df["shots_diff"] = df["home_Total Shots"] - df["away_Total Shots"]

if "home_Shots on Target" in df.columns and "away_Shots on Target" in df.columns:
    df["shots_on_target_diff"] = df["home_Shots on Target"] - df["away_Shots on Target"]
if "home_Dangerous Attacks" in df.columns and "away_Dangerous Attacks" in df.columns:
    df["dangerous_attacks_diff"] = df["home_Dangerous Attacks"] - df["away_Dangerous Attacks"]
if "home_Fouls" in df.columns and "away_Fouls" in df.columns:
    df["fouls_diff"] = df["home_Fouls"] - df["away_Fouls"]

features_finales = [
    "home_Ball Possession", "away_Ball Possession",
    "home_Total Shots", "away_Total Shots",
    "home_Shots on Target", "away_Shots on Target",
    "home_Shots off Target", "away_Shots off Target",
    "home_Blocked Shots", "away_Blocked Shots",
    "home_Fouls", "away_Fouls",
    "home_Yellow Cards", "away_Yellow Cards",
    "home_Red Cards", "away_Red Cards",
    "home_Dangerous Attacks", "away_Dangerous Attacks",
    "possession_diff", "shots_diff",
    "shots_on_target_diff", "dangerous_attacks_diff",
    "fouls_diff", "shots_per_possession_home", "shots_per_possession_away",
    "corner_efficiency_home", "corner_efficiency_away",
    "shots_efficiency_home", "shots_efficiency_away",

]
features_finales = [f for f in features_finales if f in df.columns]


print(features_finales)

X = df[features_finales]
y = df["corners_binario"]

# === RANDOMIZED SEARCH CON TimeSeriesSplit ===
# === RANDOMIZED SEARCH WITH TimeSeriesSplit ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=3)

base_model = XGBClassifier(random_state=42, eval_metric='logloss')
random_search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10, cv=tscv, random_state=42, n_jobs=-1)
random_search.fit(X, y)
modelo = random_search.best_estimator_
# SHAP
# SHAP EXPLANATION PLOTS
explainer = shap.Explainer(modelo)
shap_values = explainer(X)

shap.summary_plot(shap_values, X, plot_type="bar")
shap.summary_plot(shap_values, X)  # gráfico de dispersión

# Permutation Importance
# PERMUTATION IMPORTANCE

perm_result = permutation_importance(modelo, X, y, n_repeats=10, random_state=42)
perm_series = pd.Series(perm_result.importances_mean, index=X.columns).sort_values(ascending=False)
print(perm_series.head(15))



importancia = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)

# Mostrar top 15 variables
print(importancia.head(15))

# Gráfico de barras
plt.figure(figsize=(10, 6))
importancia.head(15).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 15 Features más importantes para el modelo")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()


# === VALIDACION FINAL ===
# === FINAL VALIDATION ===

kf = TimeSeriesSplit(n_splits=5)
accuracies = []
total_correct predictions = 0
total_matches = 0
correct predictions_por_clase = {0: 0, 1: 0}
conteo_por_clase = {0: 0, 1: 0}

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    modelo_fold = XGBClassifier(**modelo.get_params())
    modelo_fold.fit(X_train_fold, y_train_fold)
    y_pred = modelo_fold.predict(X_test_fold)

    acc = accuracy_score(y_test_fold, y_pred)
    accuracies.append(acc)

    correct predictions = np.sum(y_pred == y_test_fold.to_numpy())
    total = len(y_test_fold)
    total_correct predictions += correct predictions
    total_matches += total

    for real, pred in zip(y_test_fold, y_pred):
        conteo_por_clase[real] += 1
        if real == pred:
            correct predictions_por_clase[real] += 1

    print(f"  Fold {fold}: Accuracy = {acc*100:.2f}% ({correct predictions}/{total})")

precision_over = (correct predictions_por_clase[1] / conteo_por_clase[1]) if conteo_por_clase[1] > 0 else 0
usar_filtro_historial = precision_over < 0.10

# === CÁLCULO DE PRECISIÓN DEL MODELO AL PREDECIR ≤11 CORNERS ===
# === UNDER (≤11 CORNERS) PREDICTION PRECISION ===
all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    modelo_fold = XGBClassifier(**modelo.get_params())
    modelo_fold.fit(X_train_fold, y_train_fold)
    y_pred = modelo_fold.predict(X_test_fold)

    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred)

# Calcula precisión al predecir UNDER (clase 0)
precision_under = precision_score(all_y_true, all_y_pred, pos_label=0)
print(f"\n🎯 Precisión real del modelo al predecir ≤11 corners: {precision_under*100:.2f}%")



threshold_precision_map = {}

thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

print(f"{'Threshold %':<12} | {'# matches':<10} | {'Precisión UNDER':<17}")
print("-" * 45)

for t in thresholds:
    y_proba_all = []
    y_true_all = []

    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        modelo_fold = XGBClassifier(**modelo.get_params())
        modelo_fold.fit(X_train_fold, y_train_fold)
        y_proba = modelo_fold.predict_proba(X_test_fold)

        for prob, true_label in zip(y_proba, y_test_fold):
            prob_under = prob[0]
            if prob_under >= t:
                y_proba_all.append(0)
                y_true_all.append(true_label)

    precision = precision_score(y_true_all, y_proba_all, pos_label=0) if y_proba_all else 0.0
    threshold_precision_map[round(t, 2)] = round(precision, 4)
    print(f"{int(t*100):<12} | {len(y_proba_all):<10} | {precision*100:.2f}%")



print(f"  Accuracy promedio: {np.mean(accuracies)*100:.2f}%")
print(f"  Total correct predictions: {total_correct predictions} de {total_matches} → {total_correct predictions/total_matches*100:.2f}%")
print(f"  - Aciertos para ≤11 corners: {correct predictions_por_clase[0]} de {conteo_por_clase[0]} → {(correct predictions_por_clase[0]/conteo_por_clase[0])*100:.2f}%")
print(f"  - Aciertos para ≥12 corners: {correct predictions_por_clase[1]} de {conteo_por_clase[1]} → {precision_over*100:.2f}%")
print(f"\n🧠 History-based filter activated: {'✅ SÍ' if usar_filtro_historial else '❌ NO'}")




upcoming_matches = fixtures[(fixtures["fixture.status.short"] == "NS")]
upcoming_matches["fixture.date"] = pd.to_datetime(upcoming_matches["fixture.date"]).dt.date
upcoming_matches = upcoming_matches.sort_values(by="fixture.date").head(15)

print(f"🔍 Upcoming matches found: {len(upcoming_matches)}")

parlay_principal = []
parlay_confianza_media = []

if upcoming_matches.empty:
    
else:
    for _, row in tqdm(upcoming_matches.iterrows(), total=upcoming_matches.shape[0]):
        fixture_id = row["fixture.id"]
        home_team_name = row["teams.home.name"]
        away_team_name = row["teams.away.name"]
        date = row["fixture.date"]

        home_stats = []
        away_stats = []

        for match in reversed(historial):
            if len(home_stats) < 5:
                if match.get("home_Corner Kicks") is not None and match.get("home_team") == home_team_name:
                    home_stats.append({k: v for k, v in match.items() if "home_" in k})
                if match.get("away_Corner Kicks") is not None and match.get("away_team") == home_team_name:
                    home_stats.append({k: v for k, v in match.items() if "away_" in k})
            if len(away_stats) < 5:
                if match.get("home_Corner Kicks") is not None and match.get("home_team") == away_team_name:
                    away_stats.append({k: v for k, v in match.items() if "home_" in k})
                if match.get("away_Corner Kicks") is not None and match.get("away_team") == away_team_name:
                    away_stats.append({k: v for k, v in match.items() if "away_" in k})

            if len(home_stats) >= 5 and len(away_stats) >= 5:
                break

        if len(home_stats) < 3 or len(away_stats) < 3:
            print(f"⚠️ {home_team_name} vs {away_team_name} - Insufficient data. Skipped.")
            continue

        if usar_filtro_historial:
            if not es_equipo_under(home_stats) or not es_equipo_under(away_stats):
                print(f"❌ {home_team_name} vs {away_team_name} discarded due to historical data (muchos OVERs).")
                continue

        df_home = pd.DataFrame(home_stats).drop(columns=["home_team", "away_team"], errors='ignore')
        df_away = pd.DataFrame(away_stats).drop(columns=["home_team", "away_team"], errors='ignore')
        df_home = to_numeric_safe(df_home, df_home.columns)
        df_away = to_numeric_safe(df_away, df_away.columns)

        weights_home = np.exp(np.linspace(0, -2, num=len(df_home)))
        weights_away = np.exp(np.linspace(0, -2, num=len(df_away)))

        home_avg = df_home.apply(lambda x: weighted_average(x.dropna(), weights_home))
        away_avg = df_away.apply(lambda x: weighted_average(x.dropna(), weights_away))

        input_features = {}
        for col in features_finales:
            if col.startswith("home_"):
                input_features[col] = home_avg.get(col, 0)
            elif col.startswith("away_"):
                input_features[col] = away_avg.get(col, 0)
            elif col == "possession_diff":
                input_features[col] = home_avg.get("home_Ball Possession", 0) - away_avg.get("away_Ball Possession", 0)
            elif col == "shots_diff":
                input_features[col] = home_avg.get("home_Total Shots", 0) - away_avg.get("away_Total Shots", 0)
            elif col == "shots_on_target_diff":
                input_features[col] = home_avg.get("home_Shots on Target", 0) - away_avg.get("away_Shots on Target", 0)
            elif col == "dangerous_attacks_diff":
                input_features[col] = home_avg.get("home_Dangerous Attacks", 0) - away_avg.get("away_Dangerous Attacks", 0)
            elif col == "fouls_diff":
                input_features[col] = home_avg.get("home_Fouls", 0) - away_avg.get("away_Fouls", 0)

        
        input_features["shots_per_possession_home"] = home_avg.get("home_Total Shots", 0) / (home_avg.get("home_Ball Possession", 0) + 1)
        input_features["shots_per_possession_away"] = away_avg.get("away_Total Shots", 0) / (away_avg.get("away_Ball Possession", 0) + 1)
        input_features["corner_efficiency_home"] = home_avg.get("home_Corner Kicks", 0) / (home_avg.get("home_Dangerous Attacks", 0) + 1)
        input_features["corner_efficiency_away"] = away_avg.get("away_Corner Kicks", 0) / (away_avg.get("away_Dangerous Attacks", 0) + 1)
        input_features["shots_efficiency_home"] = home_avg.get("home_Shots on Target", 0) / (home_avg.get("home_Total Shots", 0) + 1)
        input_features["shots_efficiency_away"] = away_avg.get("away_Shots on Target", 0) / (away_avg.get("away_Total Shots", 0) + 1)

        # ✅ Filtrar solo las columnas que realmente usó el modelo
        input_features = {k: v for k, v in input_features.items() if k in features_finales}

        # Convertir a DataFrame para hacer predicción
        X_input = pd.DataFrame([input_features])
        pred_proba = modelo.predict_proba(X_input)[0]

        prob_under = round(pred_proba[0] * 100, 2)
        prob_over = round(pred_proba[1] * 100, 2)

        print(f"📅 {date} - {home_team_name} vs {away_team_name}")
        print(f"  🔮 Probabilidad UNDER 11.5: {prob_under}%")
        print(f"  🔮 Probabilidad OVER 11.5: {prob_over}%")

        if prob_under >= 80:
            
            parlay_principal.append(f"{home_team_name} vs {away_team_name} ({prob_under}%)")
        elif prob_under >= 70:
            
            parlay_confianza_media.append(f"{home_team_name} vs {away_team_name} ({prob_under}%)")
        elif prob_over >= 75:
            


if parlay_principal:
    
    for partido in parlay_principal:
        print(f"  - {partido}")
else:
    

if parlay_confianza_media:
    
    for partido in parlay_confianza_media:
        print(f"  - {partido}")
else:
    



# === FUNCION PARA AJUSTAR PROBABILIDADES SEGÚN CALIBRACIÓN ===
# === FUNCTION TO CALIBRATE PREDICTION PROBABILITIES ===
def map_to_calibrated(prob):
    thresholds_sorted = sorted(threshold_precision_map.keys(), reverse=True)
    for t in thresholds_sorted:
        if prob >= t:
            return threshold_precision_map[t]
    return 0.0

# Extraer nombre y porcentaje en float de los matches futuros
matches_prob = []
for p in parlay_principal + parlay_confianza_media:
    match = re.match(r"(.+?) \(([\d.]+)%\)", p)
    if match:
        nombre = match.group(1)
        prob = float(match.group(2)) / 100
        matches_prob.append((nombre, prob))

# Aplicar calibración
matches_calibrados = [(name, map_to_calibrated(prob)) for name, prob in matches_prob]

# Calcular combinaciones únicas (sin repetir matches)
parlays_unicos = []
usados = set()

for (n1, p1), (n2, p2) in combinations(matches_calibrados, 2):
    if n1 in usados or n2 in usados:
        continue
    combined_prob = round(p1 * p2 * 100, 2)
    if combined_prob >= 64:
        parlays_unicos.append((n1, n2, combined_prob))
        usados.add(n1)
        usados.add(n2)

# Mostrar resultado final
if parlays_unicos:
    
    for p1, p2, prob in parlays_unicos:
        print(f"  • {p1} + {p2} → {prob}%")
else:
    


# In[ ]:




