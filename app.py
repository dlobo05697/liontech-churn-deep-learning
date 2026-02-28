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
st.caption("Redes Neuronales Profundas (MLP – PyTorch) | Demo + Producción")

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
    n = 300

    frequency = np.random.poisson(4, n)
    monetary = np.random.gamma(2, 300, n)
    recency = np.random.randint(1, 60, n)
    trend = np.random.normal(0, 0.3, n)

    churn = ((recency > 30) | (trend < -0.2) | (frequency <= 1)).astype(int)

    df = pd.DataFrame({
        "CodCliente": [f"DEMO_{1000+i}" for i in range(n)],
        "frequency": frequency,
        "monetary_total": monetary,
        "ticket_avg": monetary / np.maximum(frequency, 1),
        "num_productos": np.random.randint(1, 10, n),
        "num_marcas": np.random.randint(1, 5, n),
        "recency_days": recency,
        "trend_pct": trend,
        "churn": churn
    })
    return df

def agg_mes(df):
    return df.groupby("CodCliente").agg(
        frequency=("Documento", "count"),
        monetary_total=("Monto", "sum"),
        ticket_avg=("Monto", "mean"),
        ticket_max=("Monto", "max"),
        ticket_min=("Monto", "min"),
        num_productos=("CodProducto", "nunique"),
        num_familias=("CodFamilia", "nunique"),
        num_marcas=("CodMarca", "nunique"),
    ).reset_index()

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
    st.success("Modo DEMO académico activo (no requiere archivos).")
    df_model = load_demo_dataset()
    corte_texto = "Demo"
else:
    st.info("Modo producción: requiere 3 archivos (dic–ene–feb).")

    dic_file = st.file_uploader("Archivo Diciembre (TXT/CSV)", type=["txt", "csv"])
    ene_file = st.file_uploader("Archivo Enero (TXT/CSV)", type=["txt", "csv"])
    feb_file = st.file_uploader("Archivo Febrero (TXT/CSV)", type=["txt", "csv"])

    if not (dic_file and ene_file and feb_file):
        st.warning("Carga los 3 archivos para continuar.")
        st.stop()

    df_dic = read_any_delim(dic_file)
    df_ene = read_any_delim(ene_file)
    df_feb = read_any_delim(feb_file)

    # Normalización numérica
    for df in [df_dic, df_ene, df_feb]:
        df["CodCliente"] = df["CodCliente"].astype(str)
        df["Cantidad"] = pd.to_numeric(df.get("Cantidad"), errors="coerce").fillna(0)
        df["Preciounitario"] = pd.to_numeric(df.get("Preciounitario"), errors="coerce").fillna(0)
        df["Monto"] = df["Cantidad"] * df["Preciounitario"]

    dic = agg_mes(df_dic).rename(columns=lambda x: f"dic_{x}" if x != "CodCliente" else x)
    ene = agg_mes(df_ene).rename(columns=lambda x: f"ene_{x}" if x != "CodCliente" else x)
    feb = agg_mes(df_feb).rename(columns=lambda x: f"feb_{x}" if x != "CodCliente" else x)

    df_model = dic.merge(ene, on="CodCliente", how="outer") \
                  .merge(feb, on="CodCliente", how="outer") \
                  .fillna(0)

    # Features temporales (Ene -> Feb)
    df_model["trend_abs"] = df_model["feb_monetary_total"] - df_model["ene_monetary_total"]
    df_model["trend_pct"] = df_model["trend_abs"] / (df_model["ene_monetary_total"] + 1)

    # Label churn (si Feb no compra o cae fuerte vs Ene)
    df_model["churn"] = (
        (df_model["feb_frequency"] == 0) |
        (df_model["trend_pct"] < -0.30)
    ).astype(int)

    corte_texto = "Dic→Feb"

# =========================
# INFO DATASET
# =========================
st.subheader("Dataset a nivel cliente")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Corte (recency)", corte_texto)

with st.expander("Vista rápida de clientes (sample)"):
    st.dataframe(df_model.head(20), use_container_width=True)

# =========================
# ENTRENAMIENTO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional",
    min_value=50,
    max_value=len(df_model),
    value=min(2000, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    # IDs
    client_ids = df_train["CodCliente"].values

    # X, y
    X = df_train.drop(columns=["CodCliente", "churn"])
    y = df_train["churn"].astype(int).values.reshape(-1, 1)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X_scaled, y, client_ids, test_size=0.25, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # Torch tensors
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_te = torch.tensor(X_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32)
    y_te = torch.tensor(y_te, dtype=torch.float32)

    # Modelo
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

    model = MLP(X_tr.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []

    # Entrenamiento simple (estable)
    for _ in range(15):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te), y_te).item()
            val_losses.append(val_loss)

    st.success("Entrenamiento completado.")

    # Predicciones test
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = torch.sigmoid(logits).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1 = f1_score(y_te, preds, zero_division=0)
    auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")

    st.subheader("Métricas (test)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC-AUC", "N/A" if np.isnan(auc) else f"{auc:.3f}")

    # Curvas (pérdida)
    st.subheader("Curvas de pérdida")
    st.line_chart(pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}))

    # =========================
    # TOP CLIENTES + CONTADORES
    # =========================
    st.subheader("Clientes con mayor riesgo de churn (test)")

    risk_df = pd.DataFrame({
        "CodCliente": id_te,
        "prob_churn": probs
    }).sort_values("prob_churn", ascending=False)

    # Contadores por umbral
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Riesgo ≥ 50%", int((risk_df["prob_churn"] >= 0.50).sum()))
    cc2.metric("Riesgo ≥ 60%", int((risk_df["prob_churn"] >= 0.60).sum()))
    cc3.metric("Riesgo ≥ 70%", int((risk_df["prob_churn"] >= 0.70).sum()))

    # Top 60
    top60 = risk_df.head(60).copy()
    top60.insert(0, "Ranking", range(1, len(top60) + 1))

    st.markdown("### Top 60 clientes con mayor riesgo")
    st.dataframe(top60, use_container_width=True)

    # =========================
    # EXPORT CSV
    # =========================
    st.markdown("### Exportar Top 60")
    csv = top60.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Descargar Top 60 clientes en riesgo (CSV)",
        data=csv,
        file_name="top_60_clientes_churn_liontech.csv",
        mime="text/csv"
    )
