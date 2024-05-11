### # Toolbox de Machine Learning

Este repositorio contiene una colección de funciones útiles para el manejo de tareas de Machine Learning. Estas funciones pueden ser utilizadas para diversas tareas como preprocesamiento de datos, entrenamiento de modelos, evaluación de modelos, etc.

## Contenido

- `toolbox_ML.py`: Contiene funciones para el manejo de tareas de Machine Learning.
- `prueba.ipynb`: Contiene pruebas realizadas para la demostración de las funciones sobre distintos datasets.
- `Team_Challenge_ToolBox_I.ipynb`: Contiene las pautas para las funciones a realizar.

## Uso

- Puedes utilizar estas funciones en tus proyectos de Machine Learning importando el módulo correspondiente y llamando a la funciones necesaria. Por ejemplo:

```python
from toolbox_ML.py import describe_df

df_titanic = pd.read_csv('./data/titanic.csv')
describe_df(df_titanic)

- Otra forma de utilizar este toolbox:
```python
from tooolbox_ML.py as tb

df_titanic = pd.read_csv('./data/titanic.csv')
tb.tipifica_variables(df_titanic, 4, 0.6)