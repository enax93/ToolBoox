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

                

