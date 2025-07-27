# app.py
import streamlit as st
import pandas as pd
from modelo import cargar_y_entrenar_modelo

st.set_page_config(page_title="Predicción de Precio de Venta", layout="centered")
st.title("🧠 Predicción del Sale Amount")

st.markdown("Ingrese los datos del inmueble para predecir el valor de venta:")

# Entrenamos el modelo una sola vez
try:
    modelo = cargar_y_entrenar_modelo()
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# Formulario de ingreso de datos
with st.form("formulario_prediccion"):
    assessed_value = st.number_input("Assessed Value", min_value=0.0, step=1000.0)
    area = st.number_input("Área (m²)", min_value=0.0, step=1.0)
    meses_venta = st.number_input("Meses en venta", min_value=0, step=1)
    habitaciones = st.number_input("Número de habitaciones", min_value=0, step=1)
    pisos = st.number_input("Número de pisos", min_value=1, step=1)

    property_type = st.selectbox("Tipo de propiedad", ['Casa', 'Departamento', 'Terreno', 'Otro'])
    residential_type = st.selectbox("Tipo residencial", ['Residencial', 'Comercial', 'Mixto', 'Otro'])
    town = st.selectbox("Distrito (Town)", ['Miraflores', 'Surco', 'San Isidro', 'Barranco', 'Otro'])

    submitted = st.form_submit_button("Predecir")

    if submitted:
        input_data = pd.DataFrame([{
            'Assessed Value': assessed_value,
            'area_m2': area,
            'meses_en_venta': meses_venta,
            'nro_habitaciones': habitaciones,
            'nro_pisos': pisos,
            'Property Type': property_type,
            'Residential Type': residential_type,
            'Town': town
        }])

        prediccion = modelo.predict(input_data)[0]
        st.success(f"💰 Valor estimado de venta: S/ {prediccion:,.2f}")
