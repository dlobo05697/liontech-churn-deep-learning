# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Lion Tech | Churn Prediction",
    layout="wide"
)

st.title("Predicción de Churn de Clientes – Lion Tech")
st.caption("Redes Neuronales Profundas (MLP – PyTorch)")

# =========================
# UTILIDADES
# =========================
def read_any_delim(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("latin1", errors="ignore")
    buf = io.StringIO(text)
    sample = buf.read(2048)
    buf.seek(0)
    sep = "\t" if sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(buf, sep=sep)

def load_demo_dataset():
    np.random.seed(42)
    n = 120

    frequency = np.random.poisson(4, n)
    monetary = np.random.gamma(2, 300, n)
    recency = np.random.randint(1, 60, n)
    trend = np.random.normal(0, 0.3, n)

    churn = ((recency > 30) | (trend < -0.2)).astype(int)

    return pd.DataFrame({
        "frequency": frequency,
        "monetary_total": monetary,
        "ticket_avg": monetary / np.maximum(frequency, 1),
        "num_productos": np.random.randint(1, 10, n),
        "num_marcas": np.random.randint(1, 5, n),
        "recency_days": recency,
        "trend_pct": trend,
        "churn": churn
    })

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Modo de ejecución")

modo = st.sidebar.radio(
    "Seleccione el modo:",
    ["Demostración académica", "Carga de archivos (producción)"]
)

# =========================
# CARGA DE DATOS
# =========================
if modo == "Demostración académica":
    st.success("Modo DEMO académico activo")
    df_model = load_demo_dataset()

else:
    st.subheader("Carga de archivos reales (TXT / CSV)")

    dic_file = st.file_uploader("Diciembre", type=["txt", "csv"])
    ene_file = st.file_uploader("Enero", type=["txt", "csv"])
    feb_file = st.file_uploader("Febrero", type=["txt", "csv"])

    if not (dic_file and ene_file and feb_file):
        st.info("Carga los tres archivos para continuar.")
        st.stop()

    df_dic = read_any_delim(dic_file)
    df_ene = read_any_delim(ene_file)
    df_feb = read_any_delim(feb_file)

    for df in [df_dic, df_ene, df_feb]:
        df["CodCliente"] = df["CodCliente"].astype(str)
        df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)
        df["Preciounitario"] = pd.to_numeric(df["Preciounitario"], errors="coerce").fillna(0)
        df["Monto"] = df["Cantidad"] * df["Preciounitario"]

    def agg_mes(df):
        return df.groupby("CodCliente").agg(
            frequency=("Documento", "count"),
            monetary_total=("Monto", "sum"),
            ticket_avg=("Monto", "mean"),
            num_productos=("CodProducto", "nunique"),
            num_marcas=("CodMarca", "nunique"),
        ).reset_index()

    dic = agg_mes(df_dic).rename(columns=lambda x: f"dic_{x}" if x != "CodCliente" else x)
    ene = agg_mes(df_ene).rename(columns=lambda x: f"ene_{x}" if x != "CodCliente" else x)
    feb = agg_mes(df_feb).rename(columns=lambda x: f"feb_{x}" if x != "CodCliente" else x)

    df_model = dic.merge(ene, on="CodCliente", how="outer") \
                  .merge(feb, on="CodCliente", how="outer") \
                  .fillna(0)

    df_model["trend_pct"] = (
        df_model["feb_monetary_total"] - df_model["ene_monetary_total"]
    ) / (df_model["ene_monetary_total"] + 1)

    df_model["churn"] = (
        (df_model["feb_frequency"] == 0) |
        (df_model["trend_pct"] < -0.3)
    ).astype(int)

# =========================
# INFO DATASET
# =========================
st.subheader("Dataset a nivel cliente")

c1, c2 = st.columns(2)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")

with st.expander("Vista rápida"):
    st.dataframe(df_model.head())

# =========================
# ENTRENAMIENTO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional",
    min_value=50,
    max_value=len(df_model),
    value=min(500, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["churn"], errors="ignore")
    if "CodCliente" in X.columns:
        X = X.drop(columns=["CodCliente"])

    y = df_train["churn"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(15):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    st.success("Entrenamiento completado")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, preds):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, preds):.3f}")
    m4.metric("F1", f"{f1_score(y_test, preds):.3f}")
    m5.metric("ROC-AUC", f"{roc_auc_score(y_test, probs):.3f}")
