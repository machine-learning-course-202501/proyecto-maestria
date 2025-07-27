import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --------------------------
# 🖌️ Estilo con CSS personalizado
st.markdown("""
    <style>
        .stApp {
            background-color: #0033FF;
        }
        h1, .stApp h1 {
            color: white !important;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# --------------------------
# Cargar el modelo
modelo = joblib.load('modelo_inmobiliario.pkl')

# --------------------------
# Título
st.title("Predicción del Monto de Venta de Propiedades")

# --------------------------
# Entradas del usuario
#st.subheader("🏠 Ingrese las características del inmueble")
st.markdown("<h3 style='color: white;'>🏠 Ingrese las características del inmueble</h3>", unsafe_allow_html=True)

valor_catastral = st.number_input("Valor Catastral (Assessed Value)", min_value=0)

area_m2 = st.number_input("Área en m²", min_value=0.0)

meses_en_venta = st.number_input("Meses en venta", min_value=0)
habitaciones = st.number_input("Número de habitaciones", min_value=0)
pisos = st.number_input("Número de pisos", min_value=0)

tipo_propiedad = st.selectbox("Tipo de Propiedad", ['Residential', 'Single Family'])
tipo_residencial = st.selectbox("Tipo Residencial", ['Single Family'])
ciudad = st.selectbox("Ciudad", ['Portland', 'Windham'])

# --------------------------
# Botón para predecir

st.markdown("""
    <style>
        div.stButton > button {
            background-color: #28a745; /* Verde */
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-size: 16px;
        }

        div.stButton > button:hover {
            background-color: #218838; /* Verde más oscuro al pasar el mouse */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


if st.button("Predecir"):
    entrada = pd.DataFrame([{
        'Assessed Value': valor_catastral,
        'area_m2': area_m2,
        'meses_en_venta': meses_en_venta,
        'nro_habitaciones': habitaciones,
        'nro_pisos': pisos,
        'Property Type': tipo_propiedad,
        'Residential Type': tipo_residencial,
        'Town': ciudad
    }])

    resultado = modelo.predict(entrada)
    st.markdown(f"""
       <div style='
           background-color: #FFD700;
           color: white;
           padding: 10px;
           border-radius: 10px;
           text-align: center;
           font-size: 20px;
           font-weight: bold;
        '>
          💰 Monto estimado de venta: ${int(resultado[0]):,}
        </div>
    """, unsafe_allow_html=True)

