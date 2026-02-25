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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Lion Tech | Churn (Deep Learning)",
    layout="wide"
)

st.title("Predicción de Churn (Deep Learning) – Lion Tech (Dic–Ene → Feb)")
st.caption(
    "App demostrativa (Materia 4). Entrena una MLP (PyTorch) "
    "sobre datos estructurados agregados por cliente."
)

# =========================
# MODO DE EJECUCIÓN
# =========================
st.subheader("Modo de ejecución")

modo = st.radio(
    "Seleccione el modo de uso de la aplicación:",
    ["Demostración académica", "Carga de archivos (avanzado)"],
    index=0
)

# =========================
# UTILIDADES
# =========================
def read_any_delim(uploaded_file_or_path):
    if hasattr(uploaded_file_or_path, "read"):
        raw = uploaded_file_or_path.read()
    else:
        with open(uploaded_file_or_path, "rb") as f:
            raw = f.read()

    try:
        text = raw.decode("latin1", errors="ignore")
    except Exception:
        text = raw.decode("utf-8", errors="ignore")

    buf = io.StringIO(text)
    sample = buf.read(2048)
    buf.seek(0)
    sep = "\t" if sample.count("\t") > sample.count(",") else ","
    return pd.read_csv(buf, sep=sep)


def load_demo_dataset():
    # Dataset demo académico (anonimizado y reproducible)
    np.random.seed(42)
    n = 120

    frequency = np.random.poisson(lam=4, size=n)
    monetary_total = np.random.gamma(shape=2.0, scale=300, size=n)
    ticket_avg = monetary_total / np.maximum(frequency, 1)
    ticket_max = ticket_avg * np.random.uniform(1.1, 1.6, size=n)
    ticket_min = ticket_avg * np.random.uniform(0.6, 0.9, size=n)

    num_productos = np.random.randint(1, 10, size=n)
    num_familias = np.random.randint(1, 6, size=n)
    num_marcas = np.random.randint(1, 5, size=n)

    dec = monetary_total * np.random.uniform(0.4, 0.6, size=n)
    jan = monetary_total * np.random.uniform(0.4, 0.6, size=n)

    trend_abs = jan - dec
    trend_pct = trend_abs / np.maximum(dec, 1)

    recency_days = np.random.randint(1, 60, size=n)

    churn = (
        (recency_days > 30).astype(int)
        | (frequency <= 1).astype(int)
        | (trend_pct < -0.2).astype(int)
    )

    return pd.DataFrame({
        "frequency": frequency,
        "monetary_total": monetary_total,
        "ticket_avg": ticket_avg,
        "ticket_max": ticket_max,
        "ticket_min": ticket_min,
        "num_productos": num_productos,
        "num_familias": num_familias,
        "num_marcas": num_marcas,
        "dec": dec,
        "jan": jan,
        "trend_abs": trend_abs,
        "trend_pct": trend_pct,
        "recency_days": recency_days,
        "churn": churn
    })

# =========================
# CARGA DE DATOS
# =========================
if modo == "Demostración académica":
    st.success("Modo demostración académica activo. Dataset cargado automáticamente.")
    st.info(
        "Este modo utiliza un dataset académico anonimizado y embebido "
        "para validar el modelo sin cargar archivos externos."
    )
    df_model = load_demo_dataset()

else:
    st.info("Modo avanzado: carga manual de archivos.")
    dec_file = st.file_uploader("Diciembre (TXT)", type=["txt"])
    jan_file = st.file_uploader("Enero (TXT)", type=["txt"])
    feb_file = st.file_uploader("Febrero (TXT)", type=["txt"])

    if not (dec_file and jan_file and feb_file):
        st.warning("Debe cargar los tres archivos para continuar.")
        st.stop()

    dec_df = read_any_delim(dec_file)
    jan_df = read_any_delim(jan_file)
    feb_df = read_any_delim(feb_file)

    st.error("Pipeline real de producción no incluido en el demo académico.")
    st.stop()

# =========================
# DATASET INFO
# =========================
st.subheader("Dataset a nivel cliente")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Corte (recency)", "Demo")

with st.expander("Vista rápida de datos"):
    st.dataframe(df_model.head())

# =========================
# ENTRENAMIENTO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional de clientes para acelerar demo",
    min_value=10,
    max_value=len(df_model),
    value=min(100, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["churn"])
    y = df_train["churn"].astype(int).values

    num_cols = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_cols)]
    )

    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
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

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    st.success("Entrenamiento completado.")

    st.subheader("Métricas (test)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{auc:.3f}")

    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_test, preds)
    st.dataframe(pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Real 0", "Real 1"]))

    st.subheader("Curvas de pérdida")
    st.line_chart(pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses
    }))

    st.subheader("Top clientes con mayor riesgo (test)")
    risk_df = pd.DataFrame({"prob_churn": probs}).sort_values("prob_churn", ascending=False)
    st.dataframe(risk_df.head(10))
