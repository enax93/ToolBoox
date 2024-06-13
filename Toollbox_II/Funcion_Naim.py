

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

def get_features_cat_classification(dataframe, target_col, normalize=False, mi_threshold=0):
    # Comprobaciones de entrada
    if not isinstance(dataframe, pd.DataFrame):
        print("Error: El primer argumento debe ser un DataFrame de pandas.")
        return None
    
    if target_col not in dataframe.columns:
        print(f"Error: '{target_col}' no es una columna del DataFrame.")
        return None
    
    if not pd.api.types.is_categorical_dtype(dataframe[target_col]) and not pd.api.types.is_object_dtype(dataframe[target_col]):
        print(f"Error: La columna '{target_col}' debe ser categórica.")
        return None
    
    if not isinstance(normalize, bool):
        print("Error: El argumento 'normalize' debe ser un booleano.")
        return None
    
    if not isinstance(mi_threshold, (int, float)):
        print("Error: El argumento 'mi_threshold' debe ser un número.")
        return None
    
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("Error: 'mi_threshold' debe ser un valor flotante entre 0 y 1 cuando 'normalize' es True.")
        return None
    
    # Selección de características categóricas
    cat_features = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_features = [col for col in cat_features if col != target_col]
    
    # Cálculo de la información mutua
    X = dataframe[cat_features]
    y = dataframe[target_col]
    
    mi = mutual_info_classif(X, y, discrete_features=True)
    
    if normalize:
        mi_sum = sum(mi)
        if mi_sum == 0:
            print("Error: La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi_normalized = mi / mi_sum
        selected_features = [col for col, score in zip(cat_features, mi_normalized) if score >= mi_threshold]
    else:
        selected_features = [col for col, score in zip(cat_features, mi) if score >= mi_threshold]
    
    return selected_features
