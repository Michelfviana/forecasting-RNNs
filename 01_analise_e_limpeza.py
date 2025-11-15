# michel Ferreira Viana de Carvalho 202410423
 
# --- Etapa 3: Normalização, Janelamento e Divisão Temporal ---
# (Executar somente após df_interpolated estar pronto)

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Certifique-se de que df_interpolated foi definido corretamente
try:
    features = df_interpolated.columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_interpolated),
        columns=features,
        index=df_interpolated.index,
    )

    # Parâmetros de janela
    input_window = 48  # 48 horas anteriores
    output_window = 24  # 24 horas seguintes
    target_col = "NO2(GT)"

    # Função para criar janelas supervisionadas
    def create_windows(data, input_window, output_window, target_col):
        X, y = [], []
        for i in range(len(data) - input_window - output_window + 1):
            X.append(data.iloc[i : i + input_window].values)
            y.append(
                data[target_col]
                .iloc[i + input_window : i + input_window + output_window]
                .values
            )
        return np.array(X), np.array(y)

    # Cria as janelas
    X, y = create_windows(df_scaled, input_window, output_window, target_col)

    # Divisão temporal: 70% treino, 30% teste
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Salva os conjuntos para uso posterior
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    joblib.dump(scaler, "scaler.save")
    print("Conjuntos de treino/teste e scaler salvos.")
except Exception as e:
    print(f"Erro ao processar normalização/janelamento: {e}")
# Etapa 1: Análise Exploratória dos Dados (EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",", encoding="latin1")

# Remover última coluna vazia se existir
if "Unnamed: 15" in df.columns:
    df = df.drop("Unnamed: 15", axis=1)
if "Unnamed: 14" in df.columns:
    df = df.drop("Unnamed: 14", axis=1)

# Exibir informações iniciais
print("Shape:", df.shape)
print("Colunas:", df.columns.tolist())
print("Tipos de dados:")
print(df.dtypes)
print("Amostra:")
print(df.head())

# Identificar valores ausentes (-200)
missing_counts = (df == -200).sum()
print("Valores -200 por coluna:")
print(missing_counts)

# Substituir -200 por np.nan para facilitar análise
df = df.replace(-200, np.nan)

# Plotar série temporal de NO2(GT) e CO(GT)
plt.figure(figsize=(15, 5))
plt.plot(df["NO2(GT)"], label="NO2(GT)")
plt.plot(df["CO(GT)"], label="CO(GT)", alpha=0.7)
plt.title("Série Temporal de NO2(GT) e CO(GT)")
plt.xlabel("Amostra (hora)")
plt.ylabel("Concentração")
plt.legend()
plt.tight_layout()
plt.show()

# Matriz de correlação (apenas colunas numéricas)
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm"
)
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define o caminho do arquivo
file_path = "AirQualityUCI.csv"

# Etapa 1 e 2: Carregar e Inspecionar Dados
print(f"Carregando {file_path}...")
# O dataset usa ';' como separador, ',' como decimal e '-200' para valores ausentes
try:
    df = pd.read_csv(file_path, sep=";", decimal=",", na_values="-200")
    print("Arquivo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    # Se falhar, tenta sem o 'decimal' para ver o erro
    df = pd.read_csv(file_path, sep=";")
    print("Informações do DataFrame (tentativa 2):")
    df.info()

# Inspecionar as colunas
print("\nInformações Iniciais do DataFrame (df.info()):")
df.info()

# Etapa 2: Pré-processamento e Limpeza
print("\nIniciando pré-processamento...")

# Remover colunas que parecem estar vazias (comuns neste dataset)
# O df.info() mostra que Unnamed: 15 e Unnamed: 16 são majoritariamente NaN
df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"], errors="ignore")

# Criar um índice de DateTime
# Primeiro, lidamos com linhas/valores inválidos em Date ou Time
# Se houver NaNs em Date ou Time após a carga inicial, eles precisam ser tratados
df = df.dropna(subset=["Date", "Time"])  # Remove linhas onde Date ou Time são NaN

print("Combinando colunas 'Date' e 'Time' para 'DateTime'...")
try:
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
    )
    df = df.set_index("DateTime")
    df = df.drop(columns=["Date", "Time"])
    print("Índice 'DateTime' criado com sucesso.")
except Exception as e:
    print(f"Erro ao criar DateTime: {e}")
    print("Valores problemáticos em 'Date' ou 'Time':")
    print(
        df[pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce").isna()][
            "Date"
        ].head()
    )
    print(
        df[pd.to_datetime(df["Time"], format="%H.%M.%S", errors="coerce").isna()][
            "Time"
        ].head()
    )

# Ordenar o índice para garantir a ordem temporal correta
df = df.sort_index()

# Tratar valores ausentes (NaN)
print(
    f"\nTotal de valores ausentes (NaN) antes do tratamento:\n{df.isnull().sum().sort_values(ascending=False)}"
)

# Usar interpolação linear, que é adequada para séries temporais
# Remove colunas que são inteiramente NaN (se houver)
df = df.dropna(axis=1, how="all")

print("Aplicando interpolação linear para valores ausentes...")
df_interpolated = df.interpolate(method="linear")

# Se ainda houver NaNs (ex: no início dos dados), usamos 'bfill' e 'ffill'
df_interpolated = df_interpolated.bfill().ffill()

print(
    f"\nTotal de valores ausentes (NaN) após o tratamento:\n{df_interpolated.isnull().sum()}"
)


# Salvar o DataFrame limpo
cleaned_file_path = "cleaned_air_quality.csv"
df_interpolated.to_csv(cleaned_file_path)
print(f"\nDataFrame limpo e processado salvo em: {cleaned_file_path}")

# --- Etapa 3: Normalização, Janelamento e Divisão Temporal ---
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

features = df_interpolated.columns.tolist()
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_interpolated), columns=features, index=df_interpolated.index
)

# Parâmetros de janela
input_window = 48  # 48 horas anteriores
output_window = 24  # 24 horas seguintes
target_col = "NO2(GT)"


# Função para criar janelas supervisionadas
def create_windows(data, input_window, output_window, target_col):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data.iloc[i : i + input_window].values)
        y.append(
            data[target_col]
            .iloc[i + input_window : i + input_window + output_window]
            .values
        )
    return np.array(X), np.array(y)


# Cria as janelas
X, y = create_windows(df_scaled, input_window, output_window, target_col)

# Divisão temporal: 70% treino, 30% teste
split_idx = int(0.7 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Salva os conjuntos para uso posterior
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
joblib.dump(scaler, "scaler.save")
print("Conjuntos de treino/teste e scaler salvos.")

# --- Etapa 1: Análise Exploratória (Visualização) ---

# 1. Visualizar a série temporal de NO2 (variável alvo)
print("Gerando gráfico da série temporal para NO2(GT)...")
plt.figure(figsize=(15, 7))
plt.plot(
    df_interpolated.index, df_interpolated["NO2(GT)"], label="NO2(GT)", color="blue"
)
plt.title("Concentração de Dióxido de Nitrogênio (NO2) ao longo do tempo", fontsize=16)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Concentração de NO2(GT)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("no2_time_series.png")
print("Gráfico 'no2_time_series.png' salvo.")
plt.clf()  # Limpar a figura para o próximo gráfico

# 2. Visualizar a Matriz de Correlação
print("Gerando matriz de correlação...")
plt.figure(figsize=(12, 10))
# Seleciona apenas colunas numéricas (embora todas devam ser após a carga)
numeric_df = df_interpolated.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação das Variáveis", fontsize=16)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
print("Gráfico 'correlation_matrix.png' salvo.")
