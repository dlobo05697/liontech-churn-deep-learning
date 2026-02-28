# =========================
# IMPORTS
# =========================
import io
import streamlit as st
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Lion Tech | Churn (Deep Learning)",
    layout="wide"
)

st.title("Predicción del Comportamiento de Clientes utilizando Redes Neuronales Profundas")
st.caption(
    "Aplicación demostrativa y productiva (Materia 4). "
    "Pipeline real de ingeniería de datos + MLP (PyTorch)."
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
def read_any_delim(uploaded_file):
    raw = uploaded_file.read()
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
    np.random.seed(42)
    n = 120

    frequency = np.random.poisson(4, n)
    monetary_total = np.random.gamma(2.0, 300, n)
    ticket_avg = monetary_total / np.maximum(frequency, 1)
    ticket_max = ticket_avg * np.random.uniform(1.1, 1.6, n)
    ticket_min = ticket_avg * np.random.uniform(0.6, 0.9, n)

    num_productos = np.random.randint(1, 10, n)
    num_familias = np.random.randint(1, 6, n)
    num_marcas = np.random.randint(1, 5, n)

    dec = monetary_total * np.random.uniform(0.4, 0.6, n)
    jan = monetary_total * np.random.uniform(0.4, 0.6, n)

    trend_abs = jan - dec
    trend_pct = np.where(dec > 0, trend_abs / dec, 0)

    recency_days = np.random.randint(1, 60, n)

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


def normalize_tx(df):
    df.columns = [c.strip() for c in df.columns]

    df["CodCliente"] = df["CodCliente"].astype(str).str.strip()
    df["Documento"] = df["Documento"].astype(str).str.strip()
    df["FechaEmision"] = pd.to_datetime(df["FechaEmision"], errors="coerce")

    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(0)
    df["Preciounitario"] = pd.to_numeric(
        df["Preciounitario"].astype(str).str.replace(",", "."),
        errors="coerce"
    ).fillna(0)

    df = df[df["Operacion"].str.upper() == "FACTURA"].copy()
    df["line_total"] = df["Cantidad"] * df["Preciounitario"]

    return df


def build_customer_features(dec_df, jan_df, feb_df):
    dec = normalize_tx(dec_df)
    jan = normalize_tx(jan_df)
    feb = normalize_tx(feb_df)

    def invoices(tx):
        return tx.groupby(
            ["CodCliente", "Documento", "FechaEmision"],
            as_index=False
        ).agg(
            invoice_total=("line_total", "sum"),
            prod_n=("CodProducto", "nunique"),
            fam_n=("CodFamilia", "nunique"),
            brand_n=("CodMarca", "nunique")
        )

    dec_i = invoices(dec)
    jan_i = invoices(jan)
    feb_i = invoices(feb)

    cutoff = jan_i["FechaEmision"].max()

    base = pd.concat([dec_i, jan_i])

    cust = base.groupby("CodCliente", as_index=False).agg(
        frequency=("Documento", "nunique"),
        monetary_total=("invoice_total", "sum"),
        ticket_avg=("invoice_total", "mean"),
        ticket_max=("invoice_total", "max"),
        ticket_min=("invoice_total", "min"),
        num_productos=("prod_n", "sum"),
        num_familias=("fam_n", "sum"),
        num_marcas=("brand_n", "sum"),
        last_purchase=("FechaEmision", "max")
    )

    dec_sales = dec_i.groupby("CodCliente", as_index=False)["invoice_total"].sum().rename(columns={"invoice_total": "dec"})
    jan_sales = jan_i.groupby("CodCliente", as_index=False)["invoice_total"].sum().rename(columns={"invoice_total": "jan"})

    cust = cust.merge(dec_sales, on="CodCliente", how="left").merge(jan_sales, on="CodCliente", how="left")
    cust[["dec", "jan"]] = cust[["dec", "jan"]].fillna(0)

    cust["trend_abs"] = cust["jan"] - cust["dec"]
    cust["trend_pct"] = np.where(cust["dec"] > 0, cust["trend_abs"] / cust["dec"], 0)
    cust["recency_days"] = (cutoff - cust["last_purchase"]).dt.days.clip(lower=0)

    active_feb = set(feb_i["CodCliente"].unique())
    cust["churn"] = cust["CodCliente"].apply(lambda x: 0 if x in active_feb else 1)

    cust.drop(columns=["last_purchase"], inplace=True)
    return cust, cutoff


# =========================
# CARGA DE DATOS
# =========================
cutoff_display = "Demo"

if modo == "Demostración académica":
    st.success("Modo demostración académica activo.")
    df_model = load_demo_dataset()

else:
    st.info("Modo avanzado: cargue los tres TXT.")
    dec_file = st.file_uploader("Diciembre (TXT)", type="txt")
    jan_file = st.file_uploader("Enero (TXT)", type="txt")
    feb_file = st.file_uploader("Febrero (TXT)", type="txt")

    if not (dec_file and jan_file and feb_file):
        st.stop()

    dec_df = read_any_delim(dec_file)
    jan_df = read_any_delim(jan_file)
    feb_df = read_any_delim(feb_file)

    df_model, cutoff_dt = build_customer_features(dec_df, jan_df, feb_df)
    cutoff_display = str(cutoff_dt.date())
    st.success("Pipeline de producción ejecutado correctamente.")

# =========================
# DATASET INFO
# =========================
st.subheader("Dataset a nivel cliente")
c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Corte", cutoff_display)

# =========================
# ENTRENAMIENTO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

sample_size = st.slider(
    "Muestreo opcional",
    min_value=10,
    max_value=len(df_model),
    value=min(100, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["churn"])
    y = df_train["churn"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    class MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n, 64),
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
        probs = torch.sigmoid(model(X_test)).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) == 2 else float("nan")

    st.success("Entrenamiento completado.")

    st.subheader("Métricas")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")
    m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

    st.subheader("Curvas de pérdida")
    st.line_chart(pd.DataFrame({"train": train_losses, "val": val_losses}))

    st.subheader("Top clientes con mayor riesgo")
    st.dataframe(
        pd.DataFrame({"prob_churn": probs})
        .sort_values("prob_churn", ascending=False)
        .head(10)
    )
