import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificación de Accidentes",
    page_icon="🚗",
    layout="centered",
)

# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("modelo_regresion_logistica.pkl", "rb") as f:
        obj = pickle.load(f)
    model = obj[0]        # LogisticRegression
    le    = obj[1]        # LabelEncoder  (clases: d, h, m)
    cols  = list(obj[2])  # nombres de columnas (114 features)
    return model, le, cols

model, le, COLUMNS = load_model()

# ── Mapas de opciones ────────────────────────────────────────────────────────
MESES = ["ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO",
         "JULIO","AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"]

DIAS_SEMANA = ["LUNES","MARTES","MIERCOLES","JUEVES","VIERNES","SABADO","DOMINGO"]

FRANJAS = ["DIA","TARDE","NOCHE","MADRUGADA"]

CLASES_ACC = ["'CAIDA OCUPANTE'","ATROPELLO","CHOQUE","INCENDIO","OTRO","VOLCAMIENTO"]

COLISIONANTES = ["'NO REPORTADO'","'OBJETO FIJO'","SEMOVIENTE","VEHICULO"]

AREAS = ["URBANA","RURAL"]

LOCALIZACIONES = [
    "'LOTE O PREDIO'","'PASO A NIVEL'","'PASO ELEVADO'","'PASO INFERIOR'",
    "'TRAMO DE VIA'","'VIA PEATONAL'","'VIA TRONCAL'",
    "GLORIETA","INTERSECCION","PUENTE"
]

CLIMAS = ["NORMAL","LLUVIA","NIEBLA"]

MUNICIPIOS = [
    "'LA UNION'","'NO REPORTA'","'PUEBLO RICO'","'SAN JERONIMO'","'SAN LUIS'",
    "'SAN VICENTE'","'SANTA BARBARA - ANT'","'STAFE DE ANTIOQUIA'",
    "ABEJORRAL","ALEJANDRIA","AMAGA","ANTIOQUIA","ANZA","BETANIA","BURITICA",
    "CACERES","CAICEDO","CARAMANTA","COCORNA","CONCORDIA","FREDONIA","GRANADA",
    "GUARNE","GUATAPE","ITUANGO","JARDIN","JERICO","LIBORINA","PEÑOL",
    "SANTABARBARA","SOPETRAN","TAMESIS","TITIRIBI","VALPARAISO","VENECIA"
]

ETIQUETAS = {
    "d": ("Solo Daños Materiales", "🟡", "#f0c040"),
    "h": ("Heridos",               "🟠", "#ff7b00"),
    "m": ("Víctimas Mortales",     "🔴", "#e03030"),
}

# ── Función de predicción ────────────────────────────────────────────────────
def build_input(mes, dia_mes, dia_semana, franja, clase_acc,
                colisionante, area, localizacion, clima, municipio):
    row = {col: 0 for col in COLUMNS}

    def set_col(prefix, value):
        key = f"{prefix}_{value}"
        if key in row:
            row[key] = 1

    set_col("MES",                     mes)
    set_col("DIA MES",                 str(dia_mes))
    set_col("DIA SEMANA",              dia_semana)
    set_col("FRANJA HORA",             franja)
    set_col("CLASE ACCIDENTE",         clase_acc)
    set_col("DESCRIPCION COLISIONANTE",colisionante)
    set_col("AREA ACCIDENTE",          area)
    set_col("DESCRIPCION LOCALIZACION",localizacion)
    set_col("ESTADO CLIMA",            clima)
    set_col("MUNICIPIO",               municipio)

    return pd.DataFrame([row], columns=COLUMNS)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🚗 Clasificación de Gravedad de Accidentes")
st.markdown("Completa los datos del accidente y obtén la predicción del modelo de regresión logística.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    mes        = st.selectbox("📅 Mes",          MESES)
    dia_mes    = st.slider("📆 Día del mes",      1, 31, 15)
    dia_semana = st.selectbox("📅 Día de semana", DIAS_SEMANA)
    franja     = st.selectbox("🕐 Franja horaria",FRANJAS)
    clima      = st.selectbox("🌤️ Estado del clima", CLIMAS)

with col2:
    clase_acc    = st.selectbox("💥 Clase de accidente",    CLASES_ACC)
    colisionante = st.selectbox("🚛 Descripción colisionante", COLISIONANTES)
    area         = st.selectbox("🏙️ Área del accidente",    AREAS)
    localizacion = st.selectbox("📍 Localización",          LOCALIZACIONES)
    municipio    = st.selectbox("🗺️ Municipio",             MUNICIPIOS)

st.divider()

if st.button("🔍 Predecir gravedad del accidente", use_container_width=True, type="primary"):
    X = build_input(mes, dia_mes, dia_semana, franja,
                    clase_acc, colisionante, area, localizacion, clima, municipio)

    pred_encoded = model.predict(X)[0]
    proba        = model.predict_proba(X)[0]

    # Decodificar la etiqueta
    pred_label = le.classes_[pred_encoded]
    nombre, icono, color = ETIQUETAS[pred_label]

    st.markdown(f"""
    <div style="
        background-color:{color}22;
        border-left: 6px solid {color};
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 12px;
    ">
        <h2 style="margin:0; color:{color};">{icono} {nombre}</h2>
        <p style="margin:6px 0 0; font-size:1rem; color:#444;">
            El modelo clasifica este accidente como: <strong>{nombre}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📊 Probabilidades por clase")
    prob_df = pd.DataFrame({
        "Clase": [ETIQUETAS[c][0] for c in le.classes_],
        "Probabilidad": proba
    }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    prob_df["Probabilidad (%)"] = (prob_df["Probabilidad"] * 100).round(2)
    st.dataframe(prob_df[["Clase","Probabilidad (%)"]],
                 use_container_width=True, hide_index=True)

    st.bar_chart(
        pd.DataFrame({"Probabilidad": proba},
                     index=[ETIQUETAS[c][0] for c in le.classes_])
    )

st.markdown("---")
st.caption("Modelo: Regresión Logística | Datos: Accidentes de tránsito Antioquia")
