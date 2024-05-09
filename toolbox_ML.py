## FUNCION describe_df

def describe_df(df):
    """
    Genera un resumen estadístico de un dataframe proporcionando información sobre el tipo de datos,
    porcentaje de valores nulos, valores únicos y cardinalidad de cada columna, pero con las filas
    y columnas completamente intercambiadas respecto a la versión inicial.

    Argumentos:
    df (pd.DataFrame): El dataframe a describir.

    Retorna:
    pd.DataFrame: Un nuevo dataframe con las estadísticas de cada columna del dataframe original,
                  con las estadísticas como columnas y las características de los datos como filas.
    """
    # Preparar el dataframe de resumen con filas y columnas intercambiadas
    summary = pd.DataFrame({
        'Data type': df.dtypes,
        'Percent missing (%)': df.isna().mean() * 100,
        'Unique values': df.nunique(),
        'Cardinality percent (%)': (df.nunique() / len(df)) * 100
    })

    return summary.transpose()  # Transponer el resultado 


def tipifica_variables(dataframe, umbral_categoria, umbral_continua):
    # Inicializar una lista para almacenar los resultados
    resultados = []
    
    # Iterar sobre cada columna del dataframe
    for columna in dataframe.columns:
        # Calcular la cardinalidad de la columna
        cardinalidad = dataframe[columna].nunique()
        
        # Calcular el porcentaje de cardinalidad
        porcentaje_cardinalidad = cardinalidad / len(dataframe)
        
        # Determinar el tipo de variable
        if cardinalidad == 2:
            tipo_sugerido = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo_sugerido = "Categórica"
        else:
            if porcentaje_cardinalidad >= umbral_continua:
                tipo_sugerido = "Numerica Continua"
            else:
                tipo_sugerido = "Numerica Discreta"
        
        # Agregar el resultado a la lista
        resultados.append({'nombre_variable': columna, 'tipo_sugerido': tipo_sugerido})
    
    # Convertir la lista de resultados en un DataFrame y devolverlo
    return pd.DataFrame(resultados)