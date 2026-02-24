import streamlit as st
import pandas as pd
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    # Soporta archivo subido (BytesIO) o ruta local
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
    # Dataset reducido y anonimizado para demostración académica
    data = {
        "frequency": [2, 5, 1, 7, 3, 4],
        "monetary_total": [120, 980, 45, 1500, 300, 620],
        "ticket_avg": [60, 196, 45, 214, 100, 155],
        "ticket_max": [80, 250, 45, 300, 150, 200],
        "ticket_min": [40, 150, 45, 180, 80, 120],
        "num_productos": [3, 8, 1, 10, 4, 6],
        "num_familias": [2, 4, 1, 5, 2, 3],
        "num_marcas": [2, 3, 1, 4, 2, 3],
        "dec": [60, 450, 20, 700, 140, 300],
        "jan": [60, 530, 25, 800, 160, 320],
        "trend_abs": [0, 80, 5, 100, 20, 20],
        "trend_pct": [0.0, 0.18, 0.25, 0.14, 0.14, 0.06],
        "recency_days": [15, 8, 40, 5, 20, 12],
        "churn": [1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# =========================
# CARGA DE DATOS
# =========================
if modo == "Demostración académica":
    st.success("Modo demostración académica activo. Dataset cargado automáticamente.")
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

    # ---- AQUÍ VA TU PIPELINE REAL DE INGENIERÍA DE DATOS ----
    # Debe construir un DataFrame a nivel cliente con la columna 'churn'
    # Por simplicidad, se asume que ya produces df_model
    st.error("Pipeline real no incluido en este bloque. Usa tu lógica existente.")
    st.stop()

# =========================
# DATASET INFO
# =========================
st.subheader("Dataset a nivel cliente")

col1, col2, col3 = st.columns(3)
col1.metric("Clientes", len(df_model))
col2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
col3.metric("Corte (recency)", "Demo")

with st.expander("Vista rápida de datos"):
    st.dataframe(df_model.head())

# =========================
# ENTRENAMIENTO DEL MODELO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional de clientes para acelerar demo",
    min_value=10,
    max_value=len(df_model),
    value=min(4000, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["churn"])
    y = df_train["churn"].astype(int).values

    num_cols = X.columns.tolist()
    cat_cols = []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols)
        ]
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

    train_losses = []
    val_losses = []

    for epoch in range(12):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
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

    # =========================
    # MÉTRICAS
    # =========================
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
    loss_df = pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    st.line_chart(loss_df)

    st.subheader("Top clientes con mayor riesgo (test)")
    risk_df = pd.DataFrame({
        "prob_churn": probs
    }).sort_values("prob_churn", ascending=False)
    st.dataframe(risk_df.head(10))
```
