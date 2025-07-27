# modelo.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Variables
features = ['Assessed Value', 'area_m2', 'meses_en_venta', 'nro_habitaciones', 'nro_pisos',
            'Property Type', 'Residential Type', 'Town']
target = 'Sale Amount'

def cargar_y_entrenar_modelo():
    # Leer el CSV con punto y coma
    df = pd.read_csv("dataset_inmobi.csv", delimiter=';')
    df.columns = df.columns.str.strip()  # Limpia espacios

    # LIMPIAR: convertir area_m2 de '155m2' a 155.0
    df['area_m2'] = df['area_m2'].str.replace('m2', '', regex=False).astype(float)

    # Validar columnas
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"❌ Falta columna: '{col}'")

    # Separar datos
    X = df[features]
    y = df[target]

    # Columnas categóricas
    columnas_categoricas = ['Property Type', 'Residential Type', 'Town']

    # Preprocesamiento
    preprocesador = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown='ignore'), columnas_categoricas)
    ], remainder='passthrough')

    # Pipeline
    modelo = Pipeline([
        ("preprocesador", preprocesador),
        ("regresor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Modelo entrenado correctamente. RMSE: {mse**0.5:.2f}")

    return modelo

# Ejecutar desde terminal
if __name__ == "__main__":
    modelo_entrenado = cargar_y_entrenar_modelo()
