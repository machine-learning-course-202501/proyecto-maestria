import streamlit as st
import joblib
import numpy as np
import pandas as pd
from modelo import cargar_y_entrenar_modelo  # Tu import original

# Configurar la página
st.set_page_config(
    page_title="Predictor de Precios Inmobiliarios",
    page_icon="🏠",
    layout="centered"
)

# CSS para replicar exactamente el diseño de tu imagen
st.markdown("""
    <style>
        /* Importar fuentes */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Resetear estilos de Streamlit */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
        }
        
        /* Ocultar elementos de Streamlit */
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        header { visibility: hidden; }
        footer { visibility: hidden; }
        .stToolbar { display: none; }

        /* Contenedor principal - exactamente como la imagen */
        .main-container {
            background: #ffffff;
            border-radius: 24px;
            padding: 32px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            max-width: 480px;
            margin: 40px auto;
            border: 3px solid #6366f1;
            position: relative;
        }

        /* Header exacto de la imagen */
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }

        .house-icon {
            width: 48px;
            height: 48px;
            background: #1f2937;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 16px;
            color: white;
            font-size: 24px;
        }

        .main-title {
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #1f2937;
            margin: 0;
            line-height: 1.2;
        }

        .subtitle {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 32px;
            font-weight: 400;
        }

        /* Estilos para labels exactos de la imagen */
        .custom-label {
            font-size: 14px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 8px;
            display: block;
        }

        /* Estilos para selectbox de Streamlit */
        .stSelectbox > div > div {
            background: #f9fafb !important;
            border: 1px solid #d1d5db !important;
            border-radius: 12px !important;
            min-height: 48px !important;
        }

        .stSelectbox > div > div > div {
            padding: 12px 16px !important;
            font-size: 14px !important;
            color: #374151 !important;
        }

        /* Estilos para number_input de Streamlit */
        .stNumberInput > div > div {
            background: #f9fafb !important;
            border-radius: 12px !important;
        }

        .stNumberInput > div > div > input {
            background: #f9fafb !important;
            border: 1px solid #d1d5db !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
            color: #374151 !important;
            min-height: 48px !important;
        }

        .stNumberInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
            outline: none !important;
        }

        .stSelectbox > div > div:focus-within {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
        }

        /* Ocultar labels por defecto de Streamlit */
        .stSelectbox > label,
        .stNumberInput > label {
            display: none !important;
        }

        /* Botón exacto de la imagen */
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            padding: 16px 24px !important;
            border-radius: 12px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            margin-top: 24px !important;
            min-height: 52px !important;
            transition: all 0.2s ease !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #5b21b6, #7c3aed) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
        }

        /* Resultado */
        .result-container {
            margin-top: 24px;
            padding: 20px;
            background: linear-gradient(135deg, #10b981, #34d399);
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 18px;
            font-weight: 600;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Estilos para mensajes de error */
        .stAlert {
            margin-top: 16px;
        }

        /* Responsive */
        @media (max-width: 640px) {
            .main-container {
                margin: 20px;
                padding: 24px;
                max-width: calc(100% - 40px);
            }

            .main-title {
                font-size: 20px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Contenedor principal
#st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header exacto de la imagen
st.markdown("""
    <style>
        .subtitle {
            color: white;
            font-size: 20px;
            text-align: center;
        }
    </style>
    <div class="header-container">
        <div class="house-icon">🏠</div>
        <h1 class="main-title">Predictor de Precios Inmobiliarios</h1>
    </div>
    <p class="subtitle">Obtén una estimación precisa del valor de mercado de tu propiedad</p>
""", unsafe_allow_html=True)

# Primera fila: Ciudad y Tipo de Propiedad
col1, col2 = st.columns(2)

with col1:
    st.markdown('<label class="custom-label">Ciudad:</label>', unsafe_allow_html=True)
    ciudad = st.selectbox(
        "", 
        ["Selecciona una ciudad", "Portland", "Windham"],
        key="ciudad"
    )

with col2:
    st.markdown('<label class="custom-label">Tipo de Propiedad:</label>', unsafe_allow_html=True)
    tipo_propiedad = st.selectbox(
        "",
        ["Selecciona tipo", "Residential", "Single Family"],
        key="tipo_prop"
    )

# Segunda fila: Tipo de Residencia y Área
col3, col4 = st.columns(2)

with col3:
    st.markdown('<label class="custom-label">Tipo de Residencia:</label>', unsafe_allow_html=True)
    tipo_residencial = st.selectbox(
        "",
        ["Selecciona subtipo", "Single Family"],
        key="tipo_res"
    )

with col4:
    st.markdown('<label class="custom-label">Área (m2):</label>', unsafe_allow_html=True)
    area_m2 = st.number_input(
        "",
        min_value=0.0,
        value=0.0,
        placeholder="Ejemplo: 240 m2",
        key="area"
    )

# Tercera fila: Habitaciones y Valor fiscal
col5, col6 = st.columns(2)

with col5:
    st.markdown('<label class="custom-label">Cantidad de habitaciones:</label>', unsafe_allow_html=True)
    habitaciones = st.number_input(
        "",
        min_value=0,
        value=0,
        placeholder="Ejemplo: 4",
        key="habitaciones"
    )

with col6:
    st.markdown('<label class="custom-label">Valor fiscal o de autovalúo:</label>', unsafe_allow_html=True)
    valor_catastral = st.number_input(
        "",
        min_value=0,
        value=0,
        placeholder="Ejemplo: $ 170,000",
        key="valor"
    )

# Campos adicionales necesarios para tu modelo (valores por defecto)
meses_en_venta = 6  # valor por defecto
pisos = 1  # valor por defecto

# Botón de predicción
if st.button("Predecir Precio de Mercado"):
    # Validaciones
    if ciudad == "Selecciona una ciudad":
        st.error("❌ Por favor selecciona una ciudad")
    elif tipo_propiedad == "Selecciona tipo":
        st.error("❌ Por favor selecciona el tipo de propiedad")
    elif tipo_residencial == "Selecciona subtipo":
        st.error("❌ Por favor selecciona el tipo de residencia")
    elif area_m2 <= 0:
        st.error("❌ Por favor ingresa el área en m²")
    elif habitaciones <= 0:
        st.error("❌ Por favor ingresa el número de habitaciones")
    elif valor_catastral <= 0:
        st.error("❌ Por favor ingresa el valor fiscal")
    else:
        try:
            # Intentar cargar tu modelo real
            with st.spinner('🤖 Analizando propiedades similares...'):
                try:
                    # Cargar el modelo (tu código original)
                    modelo = joblib.load('modelo_inmobiliario.pkl')
                    
                    # Crear DataFrame con los datos (tu estructura original)
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
                    
                    # Hacer la predicción
                    resultado = modelo.predict(entrada)
                    precio_predicho = int(resultado[0])
                    
                    # Mostrar resultado con el estilo exacto de tu código original
                    st.markdown(f"""
                        <div class="result-container">
                            💰 Monto estimado de venta: ${precio_predicho:,}
                        </div>
                    """, unsafe_allow_html=True)
                    
                except FileNotFoundError:
                    # Si no encuentra el modelo, usar tu función de entrenamiento
                    st.warning("⚠️ Modelo no encontrado. Entrenando nuevo modelo...")
                    
                    try:
                        # Usar tu función original para cargar y entrenar
                        modelo = cargar_y_entrenar_modelo()
                        
                        # Crear DataFrame con los datos
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
                        
                        # Hacer la predicción
                        resultado = modelo.predict(entrada)
                        precio_predicho = int(resultado[0])
                        
                        # Mostrar resultado
                        st.markdown(f"""
                            <div class="result-container">
                                💰 Monto estimado de venta: ${precio_predicho:,}
                            </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        # Fallback: estimación simulada
                        st.warning("⚠️ Error al cargar modelo. Usando estimación aproximada...")
                        
                        # Cálculo simulado mejorado
                        precio_base = area_m2 * 2500
                        factor_habitaciones = habitaciones * 25000
                        factor_ubicacion = 50000 if ciudad == "Portland" else 30000
                        factor_valor_catastral = valor_catastral * 1.2
                        
                        resultado_simulado = int((precio_base + factor_habitaciones + 
                                               factor_ubicacion + factor_valor_catastral) / 2)
                        
                        st.markdown(f"""
                            <div class="result-container">
                                💰 Monto estimado de venta: ${resultado_simulado:,}
                            </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")

# Cerrar contenedor
st.markdown('</div>', unsafe_allow_html=True)

# Información adicional (opcional)
with st.expander("ℹ️ Información sobre el modelo"):
    st.write("""
    **Características del modelo:**
    - Utiliza algoritmos de machine learning entrenados con datos inmobiliarios reales
    - Considera múltiples factores: ubicación, área, habitaciones, valor catastral
    - La predicción se basa en propiedades similares en el mercado
    
    **Nota:** Esta es una estimación basada en datos históricos. 
    El precio real puede variar según condiciones actuales del mercado.
    """)