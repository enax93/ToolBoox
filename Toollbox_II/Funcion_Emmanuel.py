import pandas as pd
import numpy as np
from scipy.stats import f_oneway

def get_features_num_classification(df, target_col, pvalue=0.05):
    """
    Identifica columnas numéricas en un DataFrame que tienen un resultado significativo
    en la prueba ANOVA con respecto a una columna objetivo categórica.

    Argumentos:
    df (DataFrame): El DataFrame de entrada que contiene los datos.
    target_col (str): El nombre de la columna objetivo en el DataFrame. Esta debe ser 
                      una columna categórica con baja cardinalidad (10 o menos valores únicos).
    pvalue (float): El nivel de significancia para la prueba ANOVA. El valor por defecto es 0.05.

    Retorna:
    list: Una lista de nombres de columnas numéricas que tienen una relación significativa con 
          la columna objetivo según la prueba ANOVA.
          Retorna None si alguna de las comprobaciones de los argumentos de entrada falla, 
          e imprime un mensaje indicando la razón.
    """
    
    # Comprobación de que el DataFrame no está vacío
    if df.empty:
        print("El DataFrame está vacío.")
        return None
    
    # Comprobación de que target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no se encuentra en el DataFrame.")
        return None
    
    # Comprobación de que target_col es categórica con baja cardinalidad
    if not pd.api.types.is_categorical_dtype(df[target_col]) and not pd.api.types.is_object_dtype(df[target_col]):
        print(f"La columna '{target_col}' no es categórica.")
        return None
    
    if df[target_col].nunique() > 10:
        print(f"La columna '{target_col}' tiene demasiadas categorías (más de 10).")
        return None
    
    # Comprobación de que pvalue es un float y está en el rango correcto
    if not isinstance(pvalue, float) or not (0 < pvalue < 1):
        print("El valor de 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Filtrar las columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Lista para almacenar las columnas que cumplen con el criterio
    significant_columns = []
    
    # Realizar el test ANOVA
    for col in numeric_cols:
        groups = [df[col][df[target_col] == category] for category in df[target_col].unique()]
        f_stat, p_val = f_oneway(*groups)
        if p_val >= (1 - pvalue):
            significant_columns.append(col)
    
    return significant_columns
