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
    f1_score, roc_auc_score, confusion_matrix
)

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Lion Tech | Churn Prediction (Producción)",
    layout="wide"
)

st.title("Predicción de Churn de Clientes – Lion Tech")
st.caption("Pipeline productivo con PyTorch + datos reales (TXT/CSV)")

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

# =========================
# SIDEBAR – MODO
# =========================
st.sidebar.title("Modo de ejecución")

modo = st.sidebar.radio(
    "Seleccione el modo de uso:",
    ["Carga de archivos (avanzado)"]
)

# =========================
# CARGA DE ARCHIVOS
# =========================
st.subheader("Carga de archivos de facturación")

dic_file = st.file_uploader("Archivo Diciembre (TXT / CSV)", type=["txt", "csv"])
ene_file = st.file_uploader("Archivo Enero (TXT / CSV)", type=["txt", "csv"])
feb_file = st.file_uploader("Archivo Febrero (TXT / CSV)", type=["txt", "csv"])

if not (dic_file and ene_file and feb_file):
    st.info("Carga los tres archivos para continuar.")
    st.stop()

df_dic = read_any_delim(dic_file)
df_ene = read_any_delim(ene_file)
df_feb = read_any_delim(feb_file)

# =========================
# NORMALIZACIÓN BÁSICA
# =========================
for df in [df_dic, df_ene, df_feb]:
    df["CodCliente"] = df["CodCliente"].astype(str)
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)
    df["Preciounitario"] = pd.to_numeric(df["Preciounitario"], errors="coerce").fillna(0)
    df["Monto"] = df["Cantidad"] * df["Preciounitario"]

# =========================
# AGREGACIÓN POR CLIENTE
# =========================
def agg_mes(df):
    return (
        df.groupby("CodCliente")
        .agg(
            frequency=("Documento", "count"),
            monetary_total=("Monto", "sum"),
            ticket_avg=("Monto", "mean"),
            ticket_max=("Monto", "max"),
            ticket_min=("Monto", "min"),
            num_productos=("CodProducto", "nunique"),
            num_familias=("CodFamilia", "nunique"),
            num_marcas=("CodMarca", "nunique"),
        )
        .reset_index()
    )

dic_agg = agg_mes(df_dic).rename(columns=lambda x: f"dic_{x}" if x != "CodCliente" else x)
ene_agg = agg_mes(df_ene).rename(columns=lambda x: f"ene_{x}" if x != "CodCliente" else x)
feb_agg = agg_mes(df_feb).rename(columns=lambda x: f"feb_{x}" if x != "CodCliente" else x)

# =========================
# DATASET FINAL
# =========================
df_model = dic_agg.merge(ene_agg, on="CodCliente", how="outer") \
                  .merge(feb_agg, on="CodCliente", how="outer") \
                  .fillna(0)

# =========================
# FEATURE ENGINEERING
# =========================
df_model["trend_abs"] = df_model["feb_monetary_total"] - df_model["ene_monetary_total"]
df_model["trend_pct"] = df_model["trend_abs"] / (df_model["ene_monetary_total"] + 1)

df_model["churn"] = (
    (df_model["feb_frequency"] == 0) |
    (df_model["trend_pct"] < -0.3)
).astype(int)

# =========================
# INFO DATASET
# =========================
st.subheader("Dataset a nivel cliente")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Ventana análisis", "Dic → Feb")

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
    value=min(1000, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["CodCliente", "churn"])
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
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
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

    train_losses, val_losses = [], []

    for _ in range(15):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    st.success("Entrenamiento completado.")

    # =========================
    # MÉTRICAS
    # =========================
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{auc:.3f}")

    # =========================
    # TOP 60 CLIENTES
    # =========================
    st.subheader("Top 60 clientes con mayor riesgo de churn")

    X_all = scaler.transform(df_model.drop(columns=["CodCliente", "churn"]))
    X_all = torch.tensor(X_all, dtype=torch.float32)

    with torch.no_grad():
        all_probs = torch.sigmoid(model(X_all)).numpy().ravel()

    risk_df = pd.DataFrame({
        "Ranking": np.arange(1, len(df_model)+1),
        "CodCliente": df_model["CodCliente"].values,
        "prob_churn": all_probs
    }).sort_values("prob_churn", ascending=False).head(60)

    st.dataframe(risk_df, use_container_width=True)

    # =========================
    # EXPORT CSV
    # =========================
    csv = risk_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Descargar Top 60 clientes en riesgo (CSV)",
        data=csv,
        file_name="top_60_clientes_churn_liontech.csv",
        mime="text/csv"
    )
