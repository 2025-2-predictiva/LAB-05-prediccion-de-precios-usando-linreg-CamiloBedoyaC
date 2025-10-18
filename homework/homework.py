#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

# ============================================================
# Paso 1: Importar librerías y cargar los datos
# ============================================================

import pandas as pd
import os
import gzip
import pickle
import json

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# Cargar datasets
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")

# ============================================================
# Paso 2: Preprocesamiento de los datos
# ============================================================

# Crear columna 'Age' a partir de 'Year' (asumiendo año actual 2021)
train_data['Age'] = 2021 - train_data['Year']
test_data['Age'] = 2021 - test_data['Year']

# Eliminar columnas innecesarias
train_data.drop(columns=['Year', 'Car_Name'], inplace=True)
test_data.drop(columns=['Year', 'Car_Name'], inplace=True)

# Separar variables independientes y dependientes
x_train = train_data.drop(columns="Present_Price")
y_train = train_data["Present_Price"]

x_test = test_data.drop(columns="Present_Price")
y_test = test_data["Present_Price"]

# ============================================================
# Paso 3: Crear pipeline de preprocesamiento y modelo
# ============================================================

# Definir columnas categóricas y numéricas
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = [col for col in x_train.columns if col not in categorical_features]

# Preprocesador de datos
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('scaler', MinMaxScaler(), numerical_features),
    ],
)

# Pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ('feature_selection', SelectKBest(f_regression)),
    ('regressor', LinearRegression())
])

# ============================================================
# Paso 4: Optimización de hiperparámetros con GridSearchCV
# ============================================================

param_grid = {
    'feature_selection__k': range(1, 15),
    'regressor__fit_intercept': [True, False],
    'regressor__positive': [True, False]
}

model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# Entrenar modelo
model.fit(x_train, y_train)

# ============================================================
# Paso 5: Guardar el modelo comprimido con gzip
# ============================================================

models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)
compressed_model_path = os.path.join(models_dir, "model.pkl.gz")

with gzip.open(compressed_model_path, "wb") as file:
    pickle.dump(model, file)

# ============================================================
# Paso 6: Calcular métricas y guardarlas en JSON
# ============================================================

def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Métricas entrenamiento
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(median_absolute_error(y_train, y_train_pred))
    }

    # Métricas prueba
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(median_absolute_error(y_test, y_test_pred)),
    }

    # Guardar en archivo JSON
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.json')

    with open(output_path, 'w') as f:
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

# Calcular y guardar métricas
calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)