# michel Ferreira Viana de Carvalho 202410423

# Forecasting de NO2 com LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 1. Carregar e Limpar o Dataset ---
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",", encoding="latin1")
# Remover colunas vazias
for col in ["Unnamed: 15", "Unnamed: 16", "Unnamed: 14"]:
    if col in df.columns:
        df = df.drop(col, axis=1)
# Substituir -200 por NaN
df = df.replace(-200, np.nan)

# --- 2. Análise Exploratória ---
print("Shape:", df.shape)
print("Colunas:", df.columns.tolist())
print("Tipos de dados:\n", df.dtypes)
print("Amostra:\n", df.head())
print("Valores -200 por coluna:\n", (df.isna().sum()))

plt.figure(figsize=(15,5))
plt.plot(df["NO2(GT)"], label="NO2(GT)")
plt.plot(df["CO(GT)"], label="CO(GT)", alpha=0.7)
plt.title("Série Temporal de NO2(GT) e CO(GT)")
plt.xlabel("Amostra (hora)")
plt.ylabel("Concentração")
plt.legend()
plt.tight_layout()
plt.savefig("no2_time_series.png")
plt.close()

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# --- 3. Pré-processamento ---
# Combinar Date e Time em DateTime, remover linhas inválidas
df = df.dropna(subset=["Date", "Time"])
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
df = df.set_index("DateTime").drop(["Date", "Time"], axis=1)
df = df.sort_index()
# Interpolação e preenchimento
print("Interpolando valores ausentes...")
df = df.interpolate(method="linear").bfill().ffill()
print("Valores ausentes após tratamento:\n", df.isna().sum())
# Salvar dados limpos
cleaned_file_path = "cleaned_air_quality.csv"
df.to_csv(cleaned_file_path)

# --- 4. Normalização e Janelamento ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.save")
features = df.columns.tolist()
input_window = 48
output_window = 24
target_col = "NO2(GT)"

def create_windows(data, input_window, output_window, target_col_idx):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i+input_window])
        y.append(data[i+input_window:i+input_window+output_window, target_col_idx])
    return np.array(X), np.array(y)

target_col_idx = features.index(target_col)
X, y = create_windows(scaled, input_window, output_window, target_col_idx)
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- 5. Modelagem, Treinamento e Avaliação ---
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dense(output_window))
model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
preds = model.predict(X_test)

def inverse_transform_NO2(scaled_seq, scaler, col_name="NO2(GT)"):
    dummy = np.zeros((scaled_seq.shape[0], len(scaler.feature_names_in_)))
    idx = list(scaler.feature_names_in_).index(col_name)
    dummy[:, idx] = scaled_seq
    inv = scaler.inverse_transform(dummy)[:, idx]
    return inv

no2_preds_inv = np.array([inverse_transform_NO2(p, scaler) for p in preds])
no2_true_inv = np.array([inverse_transform_NO2(t, scaler) for t in y_test])
rmse = np.sqrt(mean_squared_error(no2_true_inv.flatten(), no2_preds_inv.flatten()))
mae = mean_absolute_error(no2_true_inv.flatten(), no2_preds_inv.flatten())
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(12,6))
plt.plot(no2_true_inv[0], label="Real")
plt.plot(no2_preds_inv[0], label="Previsto")
plt.title("NO2(GT) - 24h (Primeira Janela do Teste)")
plt.xlabel("Hora")
plt.ylabel("Concentração NO2")
plt.legend()
plt.tight_layout()
plt.savefig("no2_real_vs_previsto.png")
plt.close()

residuos = no2_true_inv.flatten() - no2_preds_inv.flatten()
plt.figure(figsize=(12,4))
plt.plot(residuos)
plt.title("Resíduos (Real - Previsto)")
plt.xlabel("Amostra")
plt.ylabel("Resíduo")
plt.tight_layout()
plt.savefig("no2_residuos.png")
plt.close()

# --- 6. Discussão e Melhorias ---
print("\n--- Discussão dos Resultados ---")
print(f"RMSE obtido: {rmse:.2f}")
print(f"MAE obtido: {mae:.2f}")
print("\nO modelo LSTM conseguiu prever a concentração de NO2 para as próximas 24h com os erros acima.\n")
print("Limitações e possíveis melhorias:")
print("- O modelo pode ser sensível a outliers e ruídos nos dados.")
print("- A arquitetura utilizada é simples (apenas 1 camada LSTM e 1 Dense).\n  Testar arquiteturas mais profundas pode melhorar o desempenho.")
print("- O tuning de hiperparâmetros (número de neurônios, camadas, batch size, etc.) pode ser realizado para buscar melhores resultados.")
print("- Testar diferentes tamanhos de janela de entrada (ex: 24h, 72h) pode impactar a performance.")
print("- Comparar LSTM com GRU e RNN simples pode indicar a melhor arquitetura para o problema.")
print("- Incluir variáveis meteorológicas externas ou outras fontes de dados pode enriquecer o modelo.")
print("- Utilizar técnicas de regularização (Dropout, etc.) pode ajudar a evitar overfitting.")
print("\nSugestão: implemente experimentos extras para comparar arquiteturas e hiperparâmetros, e discuta os resultados no relatório final.")
