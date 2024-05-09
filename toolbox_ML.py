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

def get_features_num_regresion(dataframe, target_col, umbral_corr, pvalue= None):

    '''
    La funcion get_features_num_regresion devuelve las features para la creacion de un modelo de machine learning.

    Estas features deben ser variables numericas y disponer de una correlacón y significacion estadistica significativa con el target, definidos previamente por el usuario.
    La significacion estadistica es nula por defecto.

    Argumentos:

    .-dataframe(pandas.core.frame.DataFrame) -> un dataframe pandas sobre el que realizar el estudio
    .-target_col(str) -> la columna seleccionada como target para nuestro modelo
    .-umbral_corr(float) -> la correlacion minima exigida a una variable con el target para ser designado como feature. Debe estar comprendido entre 0 y 1
    .-pvalue(float) -> la significacion estadistica Pearson maxima exigida a una variable para ser designada como feature (generalmente 0.005). None por defecto

    Retorna:

    Lista con las columnas designadas como features para el modelo.
    Tipo lista compuesto por cadenas de texto.
    '''

    df_tipos = tipifica_variables(dataframe, umbral_categoria = 10, umbral_continua = 0.7)
    columnas_num = df_tipos.index.to_list()
    for col in columnas_num.copy():
        if (col != 'Numerica Discreta') | (col != 'Numerica Continua'):
            columnas_num.remove(col)

    if (type(umbral_corr) != float) or (umbral_corr < 0) or (umbral_corr > 1):

        print('Variable umbral_corr incorrecto.')
        return None

    elif target_col not in columnas_num:

        print('La columna seleccionada como target debe ser numerica.')
        return None

    columnas_num.remove(target_col)

    lista_features = []
    for columna in columnas_num:

        no_nulos = dataframe.dropna(subset= [target_col, columna])
        corr, pearson = pearsonr(no_nulos[target_col], no_nulos[columna])

        if pvalue != None:
            if (abs(corr) >= umbral_corr) and (pearson <= pvalue):
                lista_features.append(columna)
        else:
            if abs(corr) >= umbral_corr:
                lista_features.append(columna)
    
    return lista_features