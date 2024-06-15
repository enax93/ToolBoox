

"""
La función `get_features_cat_classification` tiene como objetivo identificar las características categóricas de un DataFrame que tienen una relación significativa con una columna de destino (`target_col`). La relación se mide utilizando la información mutua (mutual information), que es una medida de dependencia entre variables.

Argumentos de Entrada:

1. `dataframe`: El DataFrame de entrada que contiene los datos.
2. `target_col`: El nombre de la columna de destino que se utiliza para la clasificación. Debe ser una variable categórica o una variable numérica con baja cardinalidad.
3. `normalize` (por defecto es False): Un indicador booleano que especifica si se debe normalizar el valor de la información mutua.
4. `mi_threshold` (por defecto es 0): Un valor umbral para la información mutua. Solo las columnas con un valor de información mutua mayor o igual a este umbral serán devueltas.

Comportamiento de la Función:

- La función primero realiza varias comprobaciones para asegurarse de que los valores de los argumentos de entrada son adecuados. Si alguno de los valores no es adecuado, la función imprime un mensaje de error y devuelve `None`.
- Si `normalize` es `False`, la función calcula la información mutua entre cada característica categórica y la columna de destino, y devuelve una lista de características cuya información mutua es mayor o igual que `mi_threshold`.
- Si `normalize` es `True`, la función normaliza los valores de información mutua dividiéndolos por la suma total de las informaciones mutuas de las características categóricas. Luego, devuelve una lista de características cuya información mutua normalizada es mayor o igual que `mi_threshold`. Además, la función verifica que `mi_threshold` es un valor flotante entre 0 y 1, arrojando un error si no lo es.
"""

from sklearn.feature_selection import mutual_info_classif
import pandas as pd

def get_features_cat_classification(dataframe, target_col, normalize=False, mi_threshold=0.0):
    # Validar el DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame de pandas.")
        return None
    
    # Validar el nombre de la columna objetivo
    if target_col not in dataframe.columns:
        print(f"Error: '{target_col}' no es una columna del DataFrame.")
        return None
    
    # Verificar que la columna objetivo es categórica o numérica discreta de baja cardinalidad
    if not (isinstance(dataframe[target_col].dtype, pd.CategoricalDtype) or 
            (dataframe[target_col].dtype in ['int64', 'float64', 'object'] and dataframe[target_col].nunique() < 20)):
        print(f"Error: La columna '{target_col}' debe ser categórica o numérica discreta con baja cardinalidad.")
        return None
    
    # Validar que normalize es un booleano
    if not isinstance(normalize, bool):
        print("Error: El argumento 'normalize' debe ser un booleano.")
        return None
    
    # Validar que mi_threshold es un número
    if not isinstance(mi_threshold, (int, float)):
        print("Error: El argumento 'mi_threshold' debe ser un número.")
        return None
    
    # Validar el rango de mi_threshold si normalize es True
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("Error: 'mi_threshold' debe ser un valor flotante entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    # Seleccionar características categóricas y numéricas discretas
    cat_features = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_features = [col for col in cat_features if col != target_col]
    
    # Codificar características categóricas
    X = pd.get_dummies(dataframe[cat_features])
    y = dataframe[target_col].astype('category').cat.codes  # Codificar la columna objetivo como categorías numéricas

    # Cálculo de la información mutua
    mi = mutual_info_classif(X, y, discrete_features=True)
    
    # Normalización de la información mutua si se requiere
    if normalize:
        total_mi = sum(mi)
        if total_mi == 0:
            print("Error: La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi = mi / total_mi
    
    # Seleccionar características basadas en el umbral
    selected_features = [cat_features[i] for i in range(len(cat_features)) if mi[i] >= mi_threshold]
    
    return selected_features
