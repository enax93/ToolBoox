### # Toolbox de Machine Learning

Este repositorio contiene una colección de funciones útiles para el manejo de tareas de Machine Learning. Estas funciones pueden ser utilizadas para diversas tareas como preprocesamiento de datos, entrenamiento de modelos, evaluación de modelos, etc.

## Contenido

- `toolbox_ML.py`: Contiene funciones para el manejo de tareas de Machine Learning.
- `prueba.ipynb`: Contiene pruebas realizadas para la demostración de las funciones sobre distintos datasets.

## Uso

- Puedes utilizar estas funciones en tus proyectos de Machine Learning importando el módulo correspondiente y llamando a la funciones necesaria. Por ejemplo:

```python
from toolbox_ML import describe_df

df_titanic = pd.read_csv('./data/titanic.csv')
describe_df(df_titanic)

#Alternativa:
from tooolbox_ML as tb

df_pokemon = pd.read_csv('./data/pokemon.csv')
tb.tipifica_variables(df_pokemon, 4, 0.6)
