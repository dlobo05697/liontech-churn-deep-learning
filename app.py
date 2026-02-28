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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Lion Tech | Churn (Deep Learning)", layout="wide")

st.title("Predicción del Comportamiento de Clientes utilizando Redes Neuronales Profundas")
st.caption("App (Materia 4). Modo demostración + modo avanzado (producción) con TXT Dic/Ene/Feb. Modelo: MLP (PyTorch).")

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
# UTILIDADES GENERALES
# =========================
def read_any_delim(uploaded_file_or_path):
    """Lee TXT/CSV con separador desconocido (coma o tab) desde uploader o ruta."""
    if hasattr(uploaded_file_or_path, "read"):
        raw = uploaded_file_or_path.read()
    else:
        with open(uploaded_file_or_path, "rb") as f:
            raw = f.read()

    # Intentos de decodificación robustos
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
    """Dataset demo académico (anonimizado y reproducible)."""
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
# PIPELINE PRODUCCIÓN (TXT -> Features Cliente -> Churn)
# =========================
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def _find_col(df: pd.DataFrame, aliases: list[str]) -> str:
    """Encuentra columna por alias (case-insensitive). Lanza error si no existe."""
    cols = {c.strip().upper(): c for c in df.columns}
    for a in aliases:
        key = a.strip().upper()
        if key in cols:
            return cols[key]
    raise ValueError(f"No se encontró columna. Se esperaba una de: {aliases}. Disponibles: {list(df.columns)}")

def _clean_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def normalize_tx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el TXT a un esquema común.
    Requiere (con alias típicos):
      FechaEmision, CodCliente, Documento, Cantidad, Preciounitario, Operacion,
      CodProducto, CodFamilia, CodMarca
    """
    df = _norm_cols(df)

    col_fecha = _find_col(df, ["FechaEmision", "Fecha Emision", "Fecha", "FECHAEMISION", "FECHA"])
    col_cliente = _find_col(df, ["CodCliente", "CodigoCliente", "Cliente", "CODCLIENTE", "COD_CLIENTE"])
    col_doc = _find_col(df, ["Documento", "Doc", "Factura", "NroDocumento", "Nro Doc", "DOCUMENTO"])
    col_qty = _find_col(df, ["Cantidad", "Cant", "Qty", "CANTIDAD"])
    col_pu = _find_col(df, ["Preciounitario", "PrecioUnitario", "Precio", "PUnit", "PRECIOUNITARIO", "PRECIO_UNITARIO"])
    col_op = _find_col(df, ["Operacion", "Operación", "TipoOperacion", "TIPO", "OPERACION"])

    # Estos tres pueden variar más; igual los buscamos con alias razonables
    col_prod = _find_col(df, ["CodProducto", "CodigoProducto", "Producto", "CODPRODUCTO", "COD_PRODUCTO"])
    col_fam = _find_col(df, ["CodFamilia", "CodigoFamilia", "Familia", "CODFAMILIA", "COD_FAMILIA"])
    col_brand = _find_col(df, ["CodMarca", "CodigoMarca", "Marca", "CODMARCA", "COD_MARCA"])

    out = pd.DataFrame({
        "FechaEmision": df[col_fecha],
        "CodCliente": df[col_cliente],
        "Documento": df[col_doc],
        "Cantidad": df[col_qty],
        "Preciounitario": df[col_pu],
        "Operacion": df[col_op],
        "CodProducto": df[col_prod],
        "CodFamilia": df[col_fam],
        "CodMarca": df[col_brand],
    })

    out["CodCliente"] = _clean_str_series(out["CodCliente"])
    out["Documento"] = _clean_str_series(out["Documento"])
    out["Operacion"] = _clean_str_series(out["Operacion"])

    out["FechaEmision"] = pd.to_datetime(out["FechaEmision"], errors="coerce")
    out["Cantidad"] = _to_float(out["Cantidad"]).fillna(0)
    out["Preciounitario"] = _to_float(out["Preciounitario"]).fillna(0)

    out["CodProducto"] = _to_float(out["CodProducto"]).fillna(0).astype(int)
    out["CodFamilia"] = _to_float(out["CodFamilia"]).fillna(0).astype(int)
    out["CodMarca"] = _to_float(out["CodMarca"]).fillna(0).astype(int)

    # Filtra por facturas (si tu sistema usa otro valor, cámbialo aquí)
    out = out[out["Operacion"].eq("FACTURA")].copy()

    out["line_amount_usd"] = out["Cantidad"] * out["Preciounitario"]

    out = out.dropna(subset=["CodCliente", "FechaEmision", "Documento"])
    out = out[out["CodCliente"].notna()]

    return out

def invoices_from_tx(tx: pd.DataFrame) -> pd.DataFrame:
    inv = (
        tx.groupby(["CodCliente", "Documento", "FechaEmision"], as_index=False)
          .agg(
              invoice_total_usd=("line_amount_usd", "sum"),
              lines=("line_amount_usd", "size"),
              qty=("Cantidad", "sum"),
              prod_n=("CodProducto", pd.Series.nunique),
              fam_n=("CodFamilia", pd.Series.nunique),
              brand_n=("CodMarca", pd.Series.nunique),
          )
    )
    inv["invoice_total_usd"] = inv["invoice_total_usd"].clip(lower=0)
    return inv

def build_customer_features(dec_df: pd.DataFrame, jan_df: pd.DataFrame, feb_df: pd.DataFrame):
    dec_tx = normalize_tx(dec_df)
    jan_tx = normalize_tx(jan_df)
    feb_tx = normalize_tx(feb_df)

    dec_inv = invoices_from_tx(dec_tx)
    jan_inv = invoices_from_tx(jan_tx)
    feb_inv = invoices_from_tx(feb_tx)

    cutoff = jan_inv["FechaEmision"].max()
    if pd.isna(cutoff):
        raise ValueError("Enero no tiene fechas válidas para calcular recency/cutoff.")

    base_inv = pd.concat([dec_inv, jan_inv], ignore_index=True)

    cust = (
        base_inv.groupby("CodCliente", as_index=False)
        .agg(
            frequency=("Documento", "nunique"),
            monetary_total=("invoice_total_usd", "sum"),
            ticket_avg=("invoice_total_usd", "mean"),
            ticket_max=("invoice_total_usd", "max"),
            ticket_min=("invoice_total_usd", "min"),
            num_productos=("prod_n", "sum"),
            num_familias=("fam_n", "sum"),
            num_marcas=("brand_n", "sum"),
            last_purchase=("FechaEmision", "max"),
        )
    )

    dec_sales = dec_inv.groupby("CodCliente", as_index=False).agg(dec=("invoice_total_usd", "sum"))
    jan_sales = jan_inv.groupby("CodCliente", as_index=False).agg(jan=("invoice_total_usd", "sum"))

    cust = cust.merge(dec_sales, on="CodCliente", how="left").merge(jan_sales, on="CodCliente", how="left")
    cust["dec"] = cust["dec"].fillna(0.0)
    cust["jan"] = cust["jan"].fillna(0.0)

    cust["trend_abs"] = cust["jan"] - cust["dec"]
    cust["trend_pct"] = np.where(cust["dec"] > 0, cust["trend_abs"] / cust["dec"], 0.0)

    cust["recency_days"] = (cutoff - cust["last_purchase"]).dt.days.clip(lower=0)

    feb_active = set(feb_inv["CodCliente"].unique())
    cust["churn"] = cust["CodCliente"].apply(lambda c: 0 if c in feb_active else 1).astype(int)

    # Limpieza numérica
    num_cols = [
        "frequency", "monetary_total", "ticket_avg", "ticket_max", "ticket_min",
        "num_productos", "num_familias", "num_marcas",
        "dec", "jan", "trend_abs", "trend_pct", "recency_days"
    ]
    for c in num_cols:
        cust[c] = pd.to_numeric(cust[c], errors="coerce").fillna(0)

    cust = cust.drop(columns=["last_purchase"], errors="ignore")
    return cust, cutoff

# =========================
# CARGA DE DATOS
# =========================
cutoff_display = "Demo"

if modo == "Demostración académica":
    st.success("Modo demostración académica activo. Dataset cargado automáticamente.")
    st.info("Este modo utiliza un dataset académico anonimizado y embebido para validar el modelo sin cargar archivos externos.")
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

    try:
        df_model, cutoff_dt = build_customer_features(dec_df, jan_df, feb_df)
        cutoff_display = str(pd.to_datetime(cutoff_dt).date())
    except Exception as e:
        st.error(f"Error en pipeline avanzado: {e}")
        st.stop()

    st.success("Pipeline avanzado ejecutado: features a nivel cliente listos para entrenar.")

# =========================
# DATASET INFO
# =========================
st.subheader("Dataset a nivel cliente")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df_model))
c2.metric("Churn rate", f"{df_model['churn'].mean()*100:.2f}%")
c3.metric("Corte (recency)", cutoff_display)

with st.expander("Vista rápida de datos"):
    st.dataframe(df_model.head())

# =========================
# ENTRENAMIENTO DEL MODELO
# =========================
st.subheader("Entrenamiento del modelo (MLP – PyTorch)")

min_slider = 10 if len(df_model) >= 10 else 1
sample_size = st.slider(
    "Muestreo opcional de clientes para acelerar",
    min_value=min_slider,
    max_value=len(df_model),
    value=min(100, len(df_model))
)

if st.button("Entrenar modelo"):
    df_train = df_model.sample(sample_size, random_state=42)

    X = df_train.drop(columns=["churn"])
    y = df_train["churn"].astype(int).values

    num_cols = X.columns.tolist()

    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols)])
    X_proc = preprocessor.fit_transform(X)

    # Si por algún motivo una clase queda vacía, quitamos stratify
    strat = y if (len(np.unique(y)) == 2 and (y == 0).sum() >= 2 and (y == 1).sum() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.3, random_state=42, stratify=strat
    )

    X_train = torch.„Registrar(X_train, dtype=antorcha.float32)
 X_test = antorcha.tensor(X_test, dtype=antorcha.float32)
 y_train = antorcha.tensor(y_tren.remodelar(-1, 1), dtype=antorcha.float32)
 y_test = antorcha.tensor(y_prueba.remodelar(-1, 1), dtype=antorcha.float32)

    clase MLP(nn.Módulo):
        def __inicio__(yo mismo, input_dim):
            super().__inicio__()
 auto.neto = nn.Secuencial(
 nn.Lineal(entrada_dim, 64),
 nn.ReLU(),
 nn.Abandono(0,3),
 nn.Lineal(64, 32),
 nn.ReLU(),
 nn.Abandono(0,3),
 nn.Lineal(32, 1)
            )

        def adelante(yo mismo, x):
 retorno auto.neto(x)

 modelo = MLP(X_tren.forma[1])
 criterio = nn.BCEWithLogitsLoss()
 optimizador = óptimo.Adán(modelo.parámetros(), lr=0,001)

 párdidas_tren, párdidas_val = [], []

 para _ en rango(12):
 modelo.tren()
 optimizador.grado_cero()
 pérdida = criterio(modelo(X_tren), y_tren)
 pérdida.hacia atrás()
 optimizador.paso()

 modelo.eval()
 con antorcha.no_grad():
 val_pérdida = criterio(modelo(Prueba X_), y_prueba)

 pérdidas_trenes.append(pérdida.articulo())
 val_pérdidas.append(val_loss.articulo())

 modelo.eval()
 con antorcha.no_grad():
 logits = modelo(Prueba X_)
 problemas = antorcha.sigmoide(logits).numpy().ravel()
 preds = (problemas >= 0,5).astype(int)

 acc = precisión_puntuación(y_test, preds)
 prec = puntuación_precisión(prueba_y, preds, división_cero=0)
 rec = recordar_puntuación(prueba_y, preds, división_cero=0)
 f1 = puntuación_f1(prueba_y, preds, división_cero=0)
 auc = puntuación roc_auc_(y_test, problemas) si len(np.sudónico(y_prueba.numpy().ravel())) == 2 else float("nan")

 st.éxito("Entretenimiento completo.")

 st.subencabezado("Métricas (prueba)")
 m1, m2, m3, m4, m5=st.columnas(5)
 m1.métrica("Precisión", f"{acc:. . . . .3fprec:")
 m2.métrica("Precisión", f"{prec:. . . . . .3f}")
 m3.métrica(„Registrador", f"{rec:. . .3f}")
 m4.métrica("F1", f"{f1:. . .3f}")
 m5.métrica(„ROC-AUC", f"{auc:. .3f}" si no np.isnan(auc) else "N/A")

 st.subencabezado("Matriz de confusión")
 cm = confusión_matriz(y_test, preds)
 st.marco(pd.Marco de datos(cm, columnas=["Pred 0", "Pred 1"], índice=[„Real 0", "Real 1"]))

 st.subencabezado(„Curvas de pérdida")
 st.gráfico de líneas(pd.Marco de datos({"pérdida_tren": pérdidas_trenes, "val_pérdida": val_pérdidas}))

 st.subencabezado("Los principales clientes se enfrentan al alcalde Riesgo (prueba)")
 riesgo_df = pd.Marco de datos({"prob_churn": problemas}).ordenar_valores("prob_churn", ascendente=Falso)
 san marco(riesgo_df.cabeza(10))
