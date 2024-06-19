import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')


def paramns_check(df:pd.DataFrame, target_col:str, columns:list, pvalue:float) -> bool:
    '''
    Esta es una funcion de comprobacion para los parametros.

    Comprobamos que:

    .- el parametro df es un dataframe de pandas
    .- el target seleccionado es categorico, definido por un str que referencia clases, en caso de ser numerico corresponderia mapearlo a str
    .- que las columnas proporcionadas son numericas 
    .- que el pvalue es numerico y esta entre 0 y 1

    La funcion devuelve un booleano que certifica si los parametros introducidos son adecuados.
    '''
    
    try:
        if type(df) != pd.core.frame.DataFrame:
            return False
        if df[target_col].dtype != 'object':
            return False
        for col in columns:
            pd.to_numeric(df[col])
        if (float(pvalue) > 1) or (float(pvalue) < 0):
            return False
    except:
        return False
    
    return True

def plot_features_num_classification(df:pd.DataFrame, target_col:str= '', columns:list= [], pvalue:float= 0.05) -> list:
    # version con generador de indices

    '''
    Parametros:
    .- df: un dataframe de pandas
    .- target_col: el nombre de la variable target (debe ser categorica objeto/str, si contiene numeros, procede mapearla)
    .- columns: el nombre de las variables numericas del df, adjuntas en una lista (vacia por defecto)
    .- pvalue: el valor con que queremos comprobar la significancia estadistica, 0.05 por defecto

    Esta funcion cumple tras objetivos: a saber:

    1.- retorna una lista con los nombres de las features numericas que superan un test anova de significancia estadistica superior al establecido en pvalue
    2.- printa una relacion de graficas comparativas de correlacion target-variables numericas para su estudio y miniEDA
    3.- printa una relacion de graficas comparativas de colinealidad entre las distinta variables numericas para su estudio y miniEDA

    Explicamos la funcion mas en detalle a continuacion.
    '''

    paramns_ok = paramns_check(df, target_col, columns, pvalue) # comprobamos que los parametros son adecuados, si no lo son retornamos None y printamos que no lo son
    if not paramns_ok:
        print('Los parametros introduciodos son incorrectos.')
        return None

    if not columns: # si no adjuntamos lista de var numericas, cogemos todas las numericas del df
        columns = df.describe().columns.tolist()

    col_anova = [] # creamos lista vacia donde almacenaremos los nombres de var numericas que cumplen el test anova

    # a continuacion realizamos el test anova
    grps = df[target_col].unique().tolist() # almacenamo los diferentes valores posibles del target en una lista
    for feature in columns: # iteramos las var numricas
        prov_list = [] # lista provisional donde almacenaremos las series de realcion de cada var numrica con los diferentes valores del target
        
        for grp in grps:
            prov_list.append(df[df[target_col] == grp][feature]) # agregamos a la lista las series que comentabamos antes
        
        f_st, p_va = stats.f_oneway(*prov_list) # realizamos el test anova sobre la var numerica de turno (en iteracion actual) en relacion con cada valor del target y comprobamos su pvalue en funcion de su varianza
        if p_va <= pvalue: # si hay significancia estadistica recahazamos H0(medias similares) y adjuntamos el nombre de la feature a col_anova 
            col_anova.append(feature) 
    
    # empezamos con las graficas
    col_anova.insert(0, target_col) # adjuntamos el target a col_anova porque lo necesitaremos para comparar y graficar

    # creamos una primera serie de graficas relacion target(categorica) con las features numericas
    # utilizaremos subplots para reflejar cada grafica individualmente. Estos subplots son referenciados mediante arrays, importante

    q_lineas = math.ceil((len(col_anova)-1)/5) # calculamos la cantidad de lineas que compondra en la figura grafica / array (cada linea comprendera 5 subplots / columnas)

    # vamos a jugar con generadores, uno simple en realidad, no lo hemos visto en temario pero para este caso resulta de mucha utilidad
    # para movernos por los subplots de la figura grafica de turno deberemos iterar las columnas segun grafiquemos diferentes relaciones target-features
    # este generador genera los indices para el subplot
    def gen_indice():
        
        while True:
            for linea in range(100):
                for columna in range(5):
                    yield linea, columna

    contador_indice = gen_indice() # instanciamos el generador


    fig, axs = plt.subplots(q_lineas, 5, figsize=(20, 4*q_lineas)) # generamos la figura grafica con la cantidad de lineas y 5 columnas, tamño acorde a la q de lineas
    fig.suptitle('Correlación target categorico VS features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9)

    columna = 0 # comenzamos en la linea 0, primera
    indice = next(contador_indice) # primer indice [0, 0]
    # comenzamos a iterar las features que tenemos que graficas
    for feature_index in range(1, len(col_anova)): # rango 1 hata final porque la primera es el target y no queremos graficar target-target
    
        try: # presumimos que la grafica dispondra de mas de 1 linea y graficaremos en consecuencia... 
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[indice], label= i)
            axs[indice].legend()
            indice = next(contador_indice) # siguiente indice
        except IndexError: # ...si la figura solo dispone de 1 linea la graficacion dara error y graficamos en consecuencia
            for i in df[col_anova[0]].unique():     
                sns.histplot(df[df[col_anova[0]] == i][col_anova[feature_index]], kde= True, ax= axs[columna], label= i)
            axs[columna].legend()
            columna += 1 # siguiente columna
    plt.show() # mostramos la figura grafica

    # graficamos la colinealidad
    sns.pairplot(df[col_anova], hue= target_col) # pairplot para todas las features numericas que han superado la significancia estadistica
    plt.suptitle('Colinealidad features numéricas con significancia estadistica > 1-pvalue')
    plt.subplots_adjust(top=0.9) 
    plt.show() # mostramos grafica
    col_anova.remove(target_col) # quitamos el target de la lista de features que han superado el test (ya ha sido util para graficar)
    
    return col_anova # devolvemos los nombres de las features que han superado la significancia estadistica
    
    

def plot_features_cat_classification(df, target_col="", columns=[], mi_threshold=0.0, normalize=False):
    """
        Pinta las distribuciones de las columnas categoricas que pasan un threshold de informacion mutua con respecto a una columna objetivo haciendo uso de la funcion get_features_cat_classification

        Entrada: 
        - df->dataframe objetivo 
        - target_col->columna(s) objetivo, pueden ser varias
        - mi_threshold->limite usado para la comprobacion de informacion mutua de las columnas
        - normalize->booleano que indica si se ha de normalizar o no a la hora de comprobar la informacion mutua

        Salida

        - Plots de las variables que han pasado el limite de informacion mutua, representando la relacion entre esa columna y la columna objetivo
    """
    if not isinstance(df, pd.DataFrame):
        print("El dataframe proporcionado en realidad no es un dataframe")
        return None
    
    if target_col == "":
        print("Especifica una columna")
        return None
    
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no esta en el datarame")
        return None

    if not isinstance(df[target_col].dtype, pd.CategoricalDtype):
        df[target_col] = df[target_col].astype('category')
    
    if not columns:
        columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
        if target_col in columns:
            columns.remove(target_col)
    
    if not all(col in df.columns for col in columns):
        print("Comprueba que todas las columnas espeficadas esten en el dataframe")
        return None
    
    selected_columns = get_features_cat_classification(df, target_col, normalize, mi_threshold)
    
    if not selected_columns:
        print("Ninguna columna cumple con la condicion de la informacion mutua")
        return None
    
    for col in selected_columns:
        plt.figure(figsize=(10, 6))
        df.groupby([col, target_col]).size().unstack().plot(kind='bar', stacked=True)
        plt.title(f'Distribucion de {target_col} con {col}')
        plt.xlabel(col)
        plt.ylabel('Contador')
        plt.legend(title=target_col)
        plt.show()



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

        