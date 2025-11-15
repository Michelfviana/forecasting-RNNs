# Etapa 4: Modelagem, Treinamento e Avaliação com RNN (LSTM)
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Carregar dados pré-processados
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
scaler = joblib.load("scaler.save")

# Arquitetura do modelo
model = Sequential()
model.add(
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)
)
model.add(Dense(24))  # Saída para 24 horas
model.compile(optimizer="adam", loss="mse")

# Treinamento
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

# Avaliação
preds = model.predict(X_test)


# Inverter normalização para NO2(GT)
def inverse_transform_NO2(scaled, scaler, col_name="NO2(GT)"):
    dummy = np.zeros((scaled.shape[0], len(scaler.feature_names_in_)))
    idx = list(scaler.feature_names_in_).index(col_name)
    dummy[:, idx] = scaled
    inv = scaler.inverse_transform(dummy)[:, idx]
    return inv


# Para cada janela, comparar as 24h previstas com as reais
no2_preds = preds
no2_true = y_test
no2_preds_inv = np.array([inverse_transform_NO2(p, scaler) for p in no2_preds])
no2_true_inv = np.array([inverse_transform_NO2(t, scaler) for t in no2_true])

# Calcular métricas
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(no2_true_inv.flatten(), no2_preds_inv.flatten()))
mae = mean_absolute_error(no2_true_inv.flatten(), no2_preds_inv.flatten())
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Gráfico Real vs Previsto (primeira janela do teste)
plt.figure(figsize=(12, 6))
plt.plot(no2_true_inv[0], label="Real")
plt.plot(no2_preds_inv[0], label="Previsto")
plt.title("NO2(GT) - 24h (Primeira Janela do Teste)")
plt.xlabel("Hora")
plt.ylabel("Concentração NO2")
plt.legend()
plt.tight_layout()
plt.savefig("no2_real_vs_previsto.png")
plt.show()

# Gráfico de resíduos (opcional)
residuos = no2_true_inv.flatten() - no2_preds_inv.flatten()
plt.figure(figsize=(12, 4))
plt.plot(residuos)
plt.title("Resíduos (Real - Previsto)")
plt.xlabel("Amostra")
plt.ylabel("Resíduo")
plt.tight_layout()
plt.savefig("no2_residuos.png")
plt.show()
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # Para salvar o scaler

# --- 1. Definir Parâmetros ---
# Conforme o PDF e as melhores práticas
FILE_PATH = "cleaned_air_quality.csv"
TARGET_COL = "NO2(GT)"  # O que queremos prever
TRAIN_SPLIT_RATIO = 0.7  # 70% para treino
N_INPUT_HOURS = 48  # Horas de entrada (2 dias)
N_OUTPUT_HOURS = 24  # Horas de saída (prever 1 dia)

# --- 2. Carregar Dados Limpos ---
print(f"Carregando dados limpos de {FILE_PATH}...")
df = pd.read_csv(FILE_PATH, parse_dates=True, index_col="DateTime")

# Garantir que todas as colunas sejam numéricas (exceto o índice)
df = df.apply(pd.to_numeric)
print(f"Dados carregados. Total de {len(df)} observações (horas).")

# --- 3. Dividir em Treino/Teste ---
# É crucial dividir ANTES de escalar
print(
    f"Dividindo dados: {TRAIN_SPLIT_RATIO*100}% treino, {(1-TRAIN_SPLIT_RATIO)*100}% teste."
)
split_index = int(len(df) * TRAIN_SPLIT_RATIO)

df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

print(f"Observações de treino: {len(df_train)}")
print(f"Observações de teste: {len(df_test)}")

# --- 4. Normalização (Scaling) ---
print("Normalizando os dados (MinMaxScaler)...")
scaler = MinMaxScaler()

# Fit o scaler APENAS nos dados de treino
scaler.fit(df_train)

# Aplicar a transformação em ambos
scaled_train = scaler.transform(df_train)
scaled_test = scaler.transform(df_test)

# Salvar o scaler para uso futuro (para reverter as previsões)
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)
print(f"Scaler salvo em '{scaler_filename}'")

# Manter os nomes das colunas para referência
feature_columns = df.columns


# --- 5. Função de Janelamento (Windowing) ---
# Esta função cria as sequências (janelas) de entrada e saída
def create_windows(data, input_steps, output_steps, target_col_index):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        # Janela de entrada (features)
        X.append(data[i : (i + input_steps)])

        # Janela de saída (apenas a coluna alvo)
        y.append(
            data[(i + input_steps) : (i + input_steps + output_steps), target_col_index]
        )

    return np.array(X), np.array(y)


# Encontrar o índice da coluna alvo ('NO2(GT)')
try:
    target_col_index = list(feature_columns).index(TARGET_COL)
    print(f"Índice da coluna alvo '{TARGET_COL}' é: {target_col_index}")
except ValueError:
    print(f"ERRO: Coluna alvo '{TARGET_COL}' não encontrada.")
    # Adicione um 'raise' ou 'sys.exit()' se isso for um erro fatal

# --- 6. Aplicar Janelamento ---
print("Criando janelas de tempo (sequências) para treino e teste...")
X_train, y_train = create_windows(
    scaled_train, N_INPUT_HOURS, N_OUTPUT_HOURS, target_col_index
)
X_test, y_test = create_windows(
    scaled_test, N_INPUT_HOURS, N_OUTPUT_HOURS, target_col_index
)

print(f"\nFormato dos dados de Treino:")
print(f"X_train (entradas): {X_train.shape}")
print(f"y_train (saídas):   {y_train.shape}")

print(f"\nFormato dos dados de Teste:")
print(f"X_test (entradas): {X_test.shape}")
print(f"y_test (saídas):   {y_test.shape}")

# --- 7. Salvar Dados Processados ---
print("\nSalvando os arrays NumPy processados...")
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
# Salvar dados de teste originais (não escalados) para plotagem
df_test.to_csv("df_test_for_plotting.csv")

print("\n--- Preparação de Dados Concluída ---")
print(
    "Os arquivos X_train.npy, y_train.npy, X_test.npy, y_test.npy, e scaler.pkl estão prontos para a etapa de modelagem."
)

# --- Etapa 5: Discussão e Melhorias ---
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