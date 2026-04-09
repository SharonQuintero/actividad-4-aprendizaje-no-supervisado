import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. LOCALIZAR DATOS
ruta_base = os.path.dirname(__file__)
ruta_csv = os.path.join(ruta_base, '../data/transporte_grupos.csv')

try:
    # 2. CARGAR DATOS
    df = pd.read_csv(ruta_csv)
    print("--- Datos de transporte cargados para agrupamiento ---")

    # 3. CREAR EL MODELO (K-Means)
    # Queremos agrupar las estaciones en 3 tipos (clusters)
    modelo = KMeans(n_clusters=3, random_state=42)
    df['grupo_estacion'] = modelo.fit_predict(df)

    print("\nResultados del Agrupamiento (Clustering):")
    print(df)

    # 4. VISUALIZACIÓN (Para tu video)
    plt.scatter(df['pasajeros_por_hora'], df['tiempo_espera_min'], c=df['grupo_estacion'], cmap='viridis')
    plt.xlabel('Pasajeros por hora')
    plt.ylabel('Tiempo de espera (min)')
    plt.title('Agrupamiento de Estaciones de Transporte')
    plt.show()

except Exception as e:
    print(f"Error: {e}")