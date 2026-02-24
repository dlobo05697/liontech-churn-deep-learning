# Lion Tech – Churn Prediction (Deep Learning, Materia 4)

App Streamlit que entrena una **MLP (PyTorch)** para predecir **churn** usando datos agregados por cliente.

## Definición de churn
- Churn = 1: compra en dic/ene y NO compra en feb
- Churn = 0: compra en feb

## Ejecutar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar
- Streamlit Community Cloud (recomendado):
  1. Subir esta carpeta a un repo GitHub
  2. En Streamlit Cloud → Deploy → selecciona repo → `app.py`
