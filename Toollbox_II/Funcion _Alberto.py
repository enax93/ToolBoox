#### FUNCION eval_model ####

'''
Esta función debe recibir un target, unas predicciones para ese target, un argumento que determine si el problema es de regresión o clasificación y una lista de métricas:
* Si el argumento dice que el problema es de regresión, la lista de métricas debe admitir las siguientes etiquetas RMSE, MAE, MAPE, GRAPH.
* Si el argumento dice que el problema es de clasificación, la lista de métrica debe admitir, ACCURACY, PRECISION, RECALL, CLASS_REPORT, MATRIX, MATRIX_RECALL, MATRIX_PRED, 
PRECISION_X, RECALL_X. En el caso de las _X, X debe ser una etiqueta de alguna de las clases admitidas en el target.

Funcionamiento:
* Para cada etiqueta en la lista de métricas:
- RMSE, debe printar por pantalla y devolver el RMSE de la predicción contra el target.
- MAE, debe pintar por pantalla y devolver el MAE de la predicción contra el target. 
- MAPE, debe pintar por pantalla y devolver el MAPE de la predcción contra el target. Si el MAPE no se pudiera calcular la función debe avisar lanzando un error con un mensaje aclaratorio
- GRAPH, la función debe pintar una gráfica comparativa (scatter plot) del target con la predicción
- ACCURACY, pintará el accuracy del modelo contra target y lo retornará.
- PRECISION, pintará la precision media contra target y la retornará.
- RECALL, pintará la recall media contra target y la retornará.
- CLASS_REPORT, mostrará el classification report por pantalla.
- MATRIX, mostrará la matriz de confusión con los valores absolutos por casilla.
- MATRIX_RECALL, mostrará la matriz de confusión con los valores normalizados según el recall de cada fila (si usas ConfussionMatrixDisplay esto se consigue con normalize = "true")
- MATRIX_PRED, mostrará la matriz de confusión con los valores normalizados según las predicciones por columna (si usas ConfussionMatrixDisplay esto se consigue con normalize = "pred")
- PRECISION_X, mostrará la precisión para la clase etiquetada con el valor que sustituya a X (ej. PRECISION_0, mostrará la precisión de la clase 0)
- RECALL_X, mostrará el recall para la clase etiquetada co nel valor que sustituya a X (ej. RECALL_red, mostrará el recall de la clase etiquetada como "red")

NOTA1: Como puede que la función devuelva varias métricas, debe hacerlo en una tupla en el orden de aparición de la métrica en la lista que se le pasa como argumento. 
Ejemplo si la lista de entrada es ["GRAPH","RMSE","MAE"], la fución pintará la comparativa, imprimirá el RMSE y el MAE (da igual que lo haga antes de dibujar la gráfica) 
y devolverá una tupla con el (RMSE,MAE) por ese orden.
NOTA2: Una lista para clasificación puede contener varias PRECISION_X y RECALL_X, pej ["PRECISION_red","PRECISION_white","RECALL_red"] es una lista válida, 
tendrá que devolver la precisión de "red", la de "white" y el recall de "red". Si algunas de las etiquetas no existe debe arrojar ese error y detener el funcionamiento.
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, median_absolute_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def eval_model(target, predictions, problem_type, metrics):
        
        """
    Evalúa un modelo de regresión o clasificación en base a las métricas especificadas.

    Argumentos:
    - target (array-like): Valores verdaderos de los datos objetivo.
    - predictions (array-like): Valores predichos por el modelo.
    - problem_type (str): Tipo de problema, 'regression' para regresión o 'classification' para clasificación.
    - metrics (list of str): Lista de métricas a evaluar. Las métricas posibles dependen del tipo de problema.

    Retorna:
    - tuple: Tupla con los resultados de las métricas solicitadas, en el orden en que aparecen en la lista de entrada.
    """
        results = []

        if problem_type == 'regression':
            if not all(metric in ['RMSE', 'MAE', 'MAPE', 'GRAPH'] for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "RMSE", "MAE", "MAPE", "GRAPH".')
            
            for metric in metrics:
                if metric == 'RMSE':
                    rmse = np.sqrt(mean_squared_error(target, predictions))
                    print(f'RMSE: {rmse}')
                    results.append(rmse)
                elif metric == 'MAE':
                    mae = mean_squared_error(target, predictions)
                    print(f'MAE: {mae}')
                    results.append(mae)
                elif metric == 'MAPE':
                    try:
                        mape = np.mean(np.abs((target - predictions / target))) * 100
                    except ZeroDivisionError:
                         raise ValueError('No se puede calcuar el MAPE porque el target contiene valores 0')
                    print(f'MAPE: {mape}')
                    results.append(mape)
                elif metric == 'GRAPH':
                    plt.figure(figsize = (12, 8))
                    plt.scatter(target, predictions, alpha = 0.3)
                    plt.xlabel('Target')
                    plt.ylabel('Predictions')
                    plt.title('Target Vs Predictions')
                    plt.grid(True)
                    plt.show()

        elif problem_type == 'classification':
            if not all(metric.startwithc(('ACCURACY', 'PRECISION', 'RECALL', 'CLASS REPORT', 'MATRIX')) for metric in metrics):
                raise ValueError('Las metricas para regresion deben ser "ACCURACY", "PRECISION", "RECALL", "CLASS_REPORT", "MATRIX", "MATRIX_RECALL", "MATRIX_PRED" o "PRECISION_X", "RECALL_X".')
            
            for metric in metrics:
                if metric == 'ACCURACY':
                    accuracy = accuracy_score(target, predictions)
                    print(f'Accuracy: {accuracy}')
                    results.append(accuracy)
                elif metric == 'PRECISION':
                    precision = precision_score(target, predictions, average = 'macro')
                    print(f'Precision: {precision}')
                    results.append(precision)
                elif metric == 'RECALL':
                    recall = recall_score(target, predictions, average = 'macro')
                    print(f'Recall: {recall}')
                    results.append(recall)
                elif metric == 'CLASS_REPORT':
                    report = classification_report(target, predictions)
                    print('Clasification Report')
                    print(report)
                elif metric == 'MATRIX':
                    con_matrix = confusion_matrix(target, predictions)
                    print('Confusion Matrix')
                    print(con_matrix)
                    disp = ConfusionMatrixDisplay(confusion_matrix = con_matrix)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_RECALL':
                    mat_rec = confusion_matrix(target, predictions, normalize = True)
                    print('Confusion Matrix Normalize Recall')
                    print(mat_rec)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_rec)
                    disp.plot()
                    plt.show()
                elif metric == 'MATRIX_PRED':
                    mat_pred = confusion_matrix(target, predictions, normalize = 'pred')
                    print(f'Confusion Matrix Normalize Predictions')
                    print(mat_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix = mat_pred)
                    disp.plot()
                    plt.show()
                elif metric.startwith('PRECISION_'):
                    label = metric.split('_')[1]
                    try:
                        precision = precision_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Precision for class "{label}: {precision}')
                        results.append(precision)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
                elif metric.startwhith('RECALL_'):
                    label = metric.split(_)[1]
                    try:
                        recall = recall_score(target, predictions, labels = [label], average = 'micro')
                        print(f'Recall for class "{label}": {recall}')
                        results.append(recall)
                    except ValueError:
                        raise ValueError(f'La clase "{label}" no esta presente en el target.')
        else:
            raise ValueError('El tipo de problema debe ser "regression" o "clasification".')
        
        return tuple(results)


### Funcion: super_selector ###

'''
Esta función debe recibir como argumento un dataframe de features "dataset", un argumento "target_col" (que puede hacer referencia a una feature numérica o categórica) que puede ser "", 
 argumento "selectores" de tipo diccionario que puede estar vacío, y un argumento "hard_voting" como una lista vacía. 

CAUSISTICA y funcionamiento:

* Si target_col no está vacío y es un columna válidad del dataframe, la función comprobará el valor de "selectores":
    * Si "selectores" es un diccionario vacío o None:
        La fución devuelve una lista con todas las columnas del dataframe que no sean el target, tengan un valor de cardinalidad diferente del 99.99% (no sean índices) y 
        no tengan un único valor.
    * Si "selectores" no es un diccionario vacío, espera encontrar las siguientes posibles claves (y actúa en consecuencia):  
        "KBest": Tendrá como valor el número de features a seleccionar aplicando un KBest. La función debe crear una lista con las features obtenidas de emplear un SelectKBest con ANOVA.  
        "FromModel": Tendrás como valores una lista con dos elementos, el primero la instancia de un modelo de referencia y el segundo un valor entero o compatible con el argumento 
        "threshold" de SelectFromModel de sklearn. En este caso la función debe crear un a lista con las features obtenidas de aplicar un SelectFromModel con el modelo de referencia, 
        y utilizando "threshold" con el valor del segundo elemento si este no es un entero. En este caso, cuando sea un entero, usarás SelectFromModel con los argumentos 
        "max_features" igual al valor del segundo elemento y "threshold" igual a -np.inf. (Esto hace que se seleccionen "max_features" features)  
        "RFE": Tendrá como valor una tupla con tres elementos. El primero será un modelo instanciado, el segundo elemento determina el número de features a seleccionar y el tercero 
        el step a aplicar. Serán los tres argumentos del RFE de sklearn que usará la función para generar una lista de features.  
        "SFS": Tendrá como valor un tupla con 2 elementos, el modelo de referencia instanciado y el numero de featureas a alcanzar. Esta vez la función empleará un SFS para obtener 
        las lista de features seleccionadas.

* La función debe devolver tantas listas seleccionadas como claves en el diccionario de selectores y una adicional con el resultado de aplicar un hard voting a las listas obtenidas 
de aplicar el diccionario "selectores" y las que contenga "hard_voting", en caso de que "hard_voting" contenga una o más listas. La función devolverá un diccionario con 
claves equivalentes a las de selectores pero con la lista correspondiente asignada a cada clave y una adicional "hard_voting" caso de que "hard_voting" como argumento no sea una lista vacía.

Ejemplo:

```python
selectores = {
    "KBest": 5,
    "FromModel": [RandomForestClassifier(),5],
    "RFE": [LogisticRegression(),5,1]
}
super_selector(train_set_titanic, target_col = "Survived", selectores = selectores, hard_voting = ["Pclass","who","embarked_S","fare","age"])

```

Devolvera un diccionario del tipo: 
```python
{
    "KBest": [lista de features obtenidas con un SelectKBest(f_classif, k=5) con fit a train_set_titanic y target_col, sin la target_col en train_set_titanic, claro],
    "FromModel": [lista de features obtenidas de aplicar un SelecFromModel con el RandomForestClassfier y max_features = 5 y threshold = -np.inf],
    "RFE": [lista de features obtenidas de un RFE con argumentos el LogisticRegressor, n_features_to_select = 5, y step = 1],
    "hard_voting": [lista con las len(hard_voting) features con más votos entre las cuatro listas]
}
```
NOTA: Si hard_voting esta a [], la función sigue devolviendo el hard_voting pero sólo con las listas creadas internamente (si hay una sola también), 
es decir que la función siempre devuelve al menos dos listas.
'''

from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def super_selector(dataset, target_col="", selectores=None, hard_voting=[]):
    """
    Selecciona características de un dataframe basado en diferentes estrategias de selección y combina los resultados.

    Argumentos:
    - dataset (pd.DataFrame): DataFrame que contiene las características y opcionalmente la columna objetivo.
    - target_col (str): Nombre de la columna objetivo. Si es una cadena vacía, la selección se hace sin considerar una columna objetivo.
    - selectores (dict): Diccionario con las estrategias de selección. Las claves pueden ser "KBest", "FromModel", "RFE" y "SFS".
    - hard_voting (list): Lista de características para incluir en el voto fuerte.

    Retorna:
    - dict: Diccionario con las listas de características seleccionadas por cada estrategia y el resultado del voto fuerte.
    """
    
    def get_feature_names_from_indices(dataset, indices):
        """Convierte una lista de índices en nombres de columnas."""
        return dataset.columns[indices].tolist()
    
    features = dataset.drop(columns=[target_col], errors='ignore')  # Excluir target_col si existe
    results = {}

    # Selección por no ser índice y no tener cardinalidad extrema
    if target_col and target_col in dataset.columns and (selectores is None or not selectores):
        all_features = [
            col for col in features.columns 
            if dataset[col].nunique() < len(dataset) * 0.9999 and dataset[col].nunique() > 1
        ]
        results["all_features"] = all_features

    # Si target_col no está vacío y es válido
    if target_col and target_col in dataset.columns:
        y = dataset[target_col]
        
        if selectores:
            if "KBest" in selectores:
                k = selectores["KBest"]
                selector = SelectKBest(score_func=f_classif, k=k)
                selector.fit(features, y)
                selected_features = get_feature_names_from_indices(features, selector.get_support(indices=True))
                results["KBest"] = selected_features

            if "FromModel" in selectores:
                model, value = selectores["FromModel"]
                if isinstance(value, int):
                    selector = SelectFromModel(model, max_features=value, threshold=-np.inf)
                else:
                    selector = SelectFromModel(model, threshold=value)
                selector.fit(features, y)
                selected_features = get_feature_names_from_indices(features, selector.get_support(indices=True))
                results["FromModel"] = selected_features

            if "RFE" in selectores:
                model, n_features_to_select, step = selectores["RFE"]
                selector = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step)
                selector.fit(features, y)
                selected_features = get_feature_names_from_indices(features, selector.get_support(indices=True))
                results["RFE"] = selected_features

            if "SFS" in selectores:
                model, k_features = selectores["SFS"]
                selector = SFS(estimator=model, 
                               k_features=k_features, 
                               forward=True, 
                               floating=False, 
                               scoring='accuracy', 
                               cv=5)
                selector.fit(features.values, y)
                selected_features = get_feature_names_from_indices(features, list(selector.k_feature_idx_))
                results["SFS"] = selected_features

    # Aplicar voto fuerte (hard voting)
    if "all_features" in results:
        hard_voting_candidates = results["all_features"]
    else:
        hard_voting_candidates = []
        for key in results:
            hard_voting_candidates.extend(results[key])
        hard_voting_candidates = list(set(hard_voting_candidates))

    if hard_voting:
        hard_voting_candidates.extend(hard_voting)
    
    if hard_voting_candidates:
        hard_voting_count = {feature: hard_voting_candidates.count(feature) for feature in set(hard_voting_candidates)}
        sorted_features = sorted(hard_voting_count.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, count in sorted_features[:len(hard_voting_candidates)]]
        results["hard_voting"] = top_features

    return results
