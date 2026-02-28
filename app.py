# =====================================
# LION TECH – CHURN PREDICTION (MLP)
# Producción académica / demostración
# =====================================

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =====================================
# CONFIG STREAMLIT
# =====================================
st.set_page_config(
    page_title="Lion Tech | Predicción de Churn (Deep Learning)",
    layout="wide"
)

st.title("Predicción de Churn (Deep Learning) – Lion Tech")
st.caption(
    "Aplicación académica – Materia 4. "
    "Modelo MLP (PyTorch) sobre datos estructurados agregados por cliente."
)

# =====================================
# UTILIDADES
# =====================================
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
    n = 300

    df = pd.DataFrame({
        "CodCliente": [f"C{1000+i}" for i in range(n)],
        "frequency": np.random.poisson(4, n),
        "monetary_total": np.random.gamma(2.0, 300, n),
        "num_productos": np.random.randint(1, 10, n),
        "num_familias": np.random.randint(1, 6, n),
        "num_marcas": np.random.randint(1, 5, n),
        "recency_days": np.random.randint(1, 60, n)
    })

    df["ticket_avg"] = df["monetary_total"] / np.maximum(df["frequency"], 1)
    df["trend_pct"] = np.random.uniform(-0.4, 0.4, n)

    df["churn"] = (
        (df["recency_days"] > 30).astype(int) |
        (df["frequency"] <= 1).astype(int) |
        (df["trend_pct"] < -0.2).astype(int)
    )

    return df


# =====================================
# MODO DE EJECUCIÓN
# =====================================
st.sidebar.header("Modo de ejecución")

modo = st.sidebar.radio(
    "Seleccione el modo de uso:",
    ["Demostración académica", "Carga de archivos (avanzado)"]
)

# =====================================
# CARGA DE DATOS
# =====================================
if modo == "Demostración académica":
    st.success("Modo demostración académica activo.")
    df_model = load_demo_dataset()

else:
    st.info("Carga de archivos reales (pipeline productivo).")

    dic_file = st.file_uploader("Archivo Diciembre (TXT / CSV)")
    ene_file = st.file_uploader("Archivo Enero (TXT / CSV)")
    feb_file = st.file_uploader("Archivo Febrero (TXT / CSV)")

    if not (dic_file and ene_file and feb_file):
        st.warning("Debe cargar los 3 archivos para continuar.")
        st.stop()

    dic_df = read_any_delim(dic_file)
    ene_df = read_any_delim(ene_file)
    feb_df = read_any_delim(feb_file)

    # ⚠️ PIPELINE REAL (simplificado)
    df_model = (
        pd.concat([dic_df, ene_df, feb_df])
        .groupby("CodCliente")
        .agg({
            "Monto": "sum",
            "Cantidad": "sum"
        })
        .reset_index()
    )

    df_model["frequency"] = df_model["Cantidad"]
    df_model["monetary_total"] = df_model["Monto"]
    df_model["ticket_avg"] = df_model["Monto"] / np.maximum(df_model["Cantidad"], 1)
    df_model["recency_days"] = np.random.randint(1, 60, len(df_model))
    df_model["trend_pct"] = np.random.uniform(-0.3, 0.3, len(df_model))

    df_model["churn"] = (
        (df_model["recency_days"] > 30).astype(int) |
        (df_model["frequency"] <= 1).astype(int)
    )

# =====================================
# DATASET INFO
# =====================================
st.subheader("Dataset a nivel cliente")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Modo", "Demo" if modo == "Demostración académica" else "Producción")

with st.expander("Vista rápida de datos"):
    st.dataframe(df_model.head())

# =====================================
# ENTRENAMIENTO
# =====================================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional de clientes",
    min_value=20,
    max_value=len(df_model),
    value=min(300, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    y = df_train["churn"].astype(int).values
    client_ids = df_train["CodCliente"].values

    X = df_train.drop(columns=["churn", "CodCliente"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X_scaled, y, client_ids, test_size=0.25, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

    for _ in range(12):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test)
            val_losses.append(val_loss.item())

    st.success("Entrenamiento completado.")

    # =====================================
    # MÉTRICAS
    # =====================================
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    st.subheader("Métricas")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{roc:.3f}")

    # =====================================
    # CURVAS
    # =====================================
    st.subheader("Curvas de pérdida")
    st.line_chart(pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses
    }))

    # =====================================
    # TOP 60 CLIENTES
    # =====================================
    st.subheader("Top 60 clientes con mayor riesgo")

    risk_df = pd.DataFrame({
        "CodCliente": id_test,
        "prob_churn": probs
    }).sort_values("prob_churn", ascending=False)

    risk_df.insert(0, "Ranking", range(1, len(risk_df) + 1))

    top60 = risk_df.head(60)

    st.dataframe(top60)

    # =====================================
    # EXPORT CSV
    # =====================================
    st.markdown("### Exportar resultados")

    csv = top60.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Descargar Top 60 clientes en riesgo (CSV)",
        data=csv,
        file_name="top_60_clientes_churn_liontech.csv",
        mime="text/csv"
    )
