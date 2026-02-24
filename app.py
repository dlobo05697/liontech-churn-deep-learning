
import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Lion Tech | Churn (Deep Learning)", layout="wide")

st.title("Predicción de Churn (Deep Learning) — Lion Tech (Dic–Ene → Feb)")
st.caption("App demostrativa (Materia 4). Entrena una MLP (PyTorch) sobre datos estructurados agregados por cliente.")

def read_any_delim(uploaded_file_or_path):
    # Soporta: archivo subido (BytesIO) o ruta en disco.
    if hasattr(uploaded_file_or_path, "read"):
        raw = uploaded_file_or_path.read()
        buf = io.BytesIO(raw)
        header = raw[:2048].decode("latin1", errors="ignore").splitlines()[0] if raw else ""
        tabc, comc = header.count("\t"), header.count(",")
        if tabc > comc:
            return pd.read_csv(buf, sep="\t", engine="python", encoding="latin1", encoding_errors="replace")
        try:
            buf.seek(0)
            return pd.read_csv(buf, sep=None, engine="python", encoding="latin1", encoding_errors="replace")
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, sep=",", engine="python", encoding="latin1", encoding_errors="replace")
    else:
        with open(uploaded_file_or_path, "r", encoding="latin1", errors="ignore") as f:
            header = f.readline()
        tabc, comc = header.count("\t"), header.count(",")
        kwargs = dict(engine="python", encoding="latin1", encoding_errors="replace")
        if tabc > comc:
            return pd.read_csv(uploaded_file_or_path, sep="\t", **kwargs)
        try:
            return pd.read_csv(uploaded_file_or_path, sep=None, **kwargs)
        except Exception:
            return pd.read_csv(uploaded_file_or_path, sep=",", **kwargs)

def prep(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in ["FechaEmision","FechaCOntable"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    def clean_id(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().strip('"').strip()
        return s if s else np.nan
    rif = df["ClienteRIF"].map(clean_id) if "ClienteRIF" in df.columns else pd.Series([np.nan]*len(df))
    cod = df["CodCliente"].map(clean_id) if "CodCliente" in df.columns else pd.Series([np.nan]*len(df))
    df["customer_id"] = rif.fillna(cod)
    for c in ["Cantidad","Preciounitario","CostoDolar","CostoBs","FactorCambio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Cantidad" in df.columns and "Preciounitario" in df.columns:
        df["line_total"] = df["Cantidad"].fillna(0) * df["Preciounitario"].fillna(0)
    else:
        df["line_total"] = 0.0
    if "FechaEmision" in df.columns:
        df = df[df["FechaEmision"].notna()]
    df = df[df["customer_id"].notna()]
    return df

def top_mode(s):
    s = s.dropna().astype(str)
    if s.empty: return np.nan
    return s.value_counts().idxmax()

def build_customer_dataset(dec_df, jan_df, feb_df):
    dec_df = prep(dec_df); dec_df["month"]="dec"
    jan_df = prep(jan_df); jan_df["month"]="jan"
    feb_df = prep(feb_df); feb_df["month"]="feb"

    hist = pd.concat([dec_df, jan_df], ignore_index=True)
    hist["Documento"] = hist["Documento"].astype(str) if "Documento" in hist.columns else ""
    hist["invoice_key"] = hist["customer_id"].astype(str) + "|" + hist["Documento"]

    invoice = hist.groupby(["customer_id","invoice_key"], as_index=False).agg(
        invoice_total=("line_total","sum"),
        invoice_date=("FechaEmision","max"),
    )
    cust = invoice.groupby("customer_id", as_index=False).agg(
        frequency=("invoice_key","nunique"),
        monetary_total=("invoice_total","sum"),
        ticket_avg=("invoice_total","mean"),
        ticket_max=("invoice_total","max"),
        ticket_min=("invoice_total","min"),
        last_purchase=("invoice_date","max"),
    )

    div = hist.groupby("customer_id", as_index=False).agg(
        num_productos=("CodProducto","nunique") if "CodProducto" in hist.columns else ("line_total","size"),
        num_familias=("Familia","nunique") if "Familia" in hist.columns else ("line_total","size"),
        num_marcas=("Marca","nunique") if "Marca" in hist.columns else ("line_total","size"),
    )
    cust = cust.merge(div, on="customer_id", how="left")

    zona_col = "ZonaDescripcion" if "ZonaDescripcion" in hist.columns else ("ZonaCLiente" if "ZonaCLiente" in hist.columns else None)
    vend_col = "Vendedor" if "Vendedor" in hist.columns else None

    cust["zona_principal"] = hist.groupby("customer_id")[zona_col].apply(top_mode).values if zona_col else np.nan
    cust["vendedor_principal"] = hist.groupby("customer_id")[vend_col].apply(top_mode).values if vend_col else np.nan

    month_tot = hist.groupby(["customer_id","month"], as_index=False).agg(monthly_total=("line_total","sum"))
    pivot = month_tot.pivot_table(index="customer_id", columns="month", values="monthly_total", fill_value=0).reset_index()
    for m in ["dec","jan"]:
        if m not in pivot.columns: pivot[m]=0.0
    pivot["trend_abs"] = pivot["jan"] - pivot["dec"]
    pivot["trend_pct"] = np.where(pivot["dec"]>0, (pivot["jan"]-pivot["dec"])/pivot["dec"], 0.0)
    cust = cust.merge(pivot[["customer_id","dec","jan","trend_abs","trend_pct"]], on="customer_id", how="left")

    cutoff = jan_df["FechaEmision"].max() if not jan_df.empty else hist["FechaEmision"].max()
    cust["recency_days"] = (cutoff - cust["last_purchase"]).dt.days.fillna(0).astype(int)

    feb_customers = set(feb_df["customer_id"].unique())
    cust["churn"] = cust["customer_id"].apply(lambda x: 0 if x in feb_customers else 1).astype(int)

    return cust, cutoff

class MLPLogits(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)

def train_mlp(X, y, num_cols, cat_cols, sample_customers=None, seed=42):
    if sample_customers is not None and sample_customers < len(X):
        X = X.sample(sample_customers, random_state=seed)
        y = y.loc[X.index]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    def to_dense(m):
        return m.toarray().astype(np.float32) if hasattr(m, "toarray") else m.astype(np.float32)

    Xtr = torch.tensor(to_dense(X_train_t))
    Xva = torch.tensor(to_dense(X_val_t))
    Xte = torch.tensor(to_dense(X_test_t))

    ytr = torch.tensor(y_train.values.reshape(-1,1).astype(np.float32))
    yva = torch.tensor(y_val.values.reshape(-1,1).astype(np.float32))

    model = MLPLogits(Xtr.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    best_state = None
    patience = 4
    pat = 0

    train_losses, val_losses = [], []
    for epoch in range(12):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            vloss = float(criterion(model(Xva), yva).item())
        val_losses.append(vloss)

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(Xte)).numpy().ravel()

    y_true = y_test.values.astype(int)
    y_pred = (proba >= 0.5).astype(int)

    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "epochs_ran": len(train_losses)
    }
    return model, preprocessor, m, (train_losses, val_losses), (y_true, y_pred, proba), X_test.index

st.sidebar.header("Archivos")
use_local = st.sidebar.checkbox("Usar archivos locales ya cargados en el servidor", value=True)
st.sidebar.caption("Si desmarcas, podrás subir tus TXT manualmente.")

if use_local:
    dec_path = "CONSULTALION_dic.TXT"
    jan_path = "99fe6ed1-7ed0-44c1-b331-9eb34bd4d5dd.TXT"
    feb_path = "95fe4c80-1da4-4bc0-8ac8-f6ccad095be2.TXT"
    dec_df = read_any_delim(dec_path)
    jan_df = read_any_delim(jan_path)
    feb_df = read_any_delim(feb_path)
    st.sidebar.success("Cargados desde disco (demo).")
else:
    dec_u = st.sidebar.file_uploader("Diciembre (TXT)", type=["txt","csv"])
    jan_u = st.sidebar.file_uploader("Enero (TXT)", type=["txt","csv"])
    feb_u = st.sidebar.file_uploader("Febrero (TXT)", type=["txt","csv"])
    if not (dec_u and jan_u and feb_u):
        st.info("Sube los 3 archivos para continuar.")
        st.stop()
    dec_df = read_any_delim(dec_u)
    jan_df = read_any_delim(jan_u)
    feb_df = read_any_delim(feb_u)

cust, cutoff = build_customer_dataset(dec_df, jan_df, feb_df)

st.subheader("Dataset a nivel cliente")
col1, col2, col3 = st.columns(3)
col1.metric("Clientes (dic+ene)", f"{len(cust):,}")
churn_rate = float(cust["churn"].mean()) if len(cust) else 0.0
col2.metric("Churn rate", f"{churn_rate:.2%}")
col3.metric("Corte (recency)", str(cutoff.date()) if pd.notna(cutoff) else "-")

with st.expander("Vista rápida de clientes (sample)"):
    st.dataframe(cust.sample(min(20, len(cust)), random_state=42))

st.subheader("Entrenamiento del modelo (MLP - PyTorch)")
sample_n = st.slider("Muestreo opcional de clientes para acelerar demo", min_value=0, max_value=20000, value=4000, step=500)
sample_n = None if sample_n == 0 else sample_n

num_cols = ["frequency","monetary_total","ticket_avg","ticket_max","ticket_min",
            "num_productos","num_familias","num_marcas","dec","jan","trend_abs","trend_pct","recency_days"]
cat_cols = ["zona_principal","vendedor_principal"]

X = cust[num_cols + cat_cols].copy()
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
y = cust["churn"].astype(int)

if st.button("Entrenar modelo"):
    with st.spinner("Entrenando..."):
        model, preproc, m, losses, pred_pack, idx_test = train_mlp(X, y, num_cols, cat_cols, sample_customers=sample_n)
    st.success("Entrenamiento completado.")

    st.markdown("### Métricas (test)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1", f"{m['f1']:.3f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if str(m["roc_auc"])!="nan" else "N/A")

    st.markdown("### Matriz de confusión")
    st.write(np.array(m["confusion_matrix"]))

    train_losses, val_losses = losses
    st.markdown("### Curvas de pérdida")
    st.line_chart(pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}))

    y_true, y_pred, proba = pred_pack
    out = pd.DataFrame({
        "customer_id": cust.loc[idx_test, "customer_id"].values,
        "churn_true": y_true,
        "churn_pred": y_pred,
        "churn_proba": proba
    }).sort_values("churn_proba", ascending=False)

    st.markdown("### Top clientes con mayor riesgo (test)")
    st.dataframe(out.head(50))

    # CSV download (lightweight)
    st.download_button(
        "Descargar predicciones (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_churn.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Definición de churn: clientes con compras en dic/ene que NO vuelven a comprar en febrero (ventana de observación).")
