<a name="readme-top"></a>
![Static Badge](https://img.shields.io/badge/Data_Science-UdeA-green)
![Static Badge](https://img.shields.io/badge/Python-3.11.6-orange)
![GitHub User's stars](https://img.shields.io/github/stars/MoisesGuerreroUdeA?style=social)

<br/>
<div align='center'>
    <h1>Diseño de una solución para la predicción a corto plazo de radiación solar en la región de la Comunidad de Castilla y León, España, para la gestión de proyectos de generación fotovoltaica</h1>
    <h2> Especialización en Analítica y Ciencia de Datos </h2>
    <p>
        Material correspondiente al espacio de trabajo, analítica y documentación relacionada con la entrega de la monografía de la Especialización en Analítica y Ciencia de Datos.
        <br/>
        <a href='docs/'><strong>Revisar la documentación »</strong></a>
    </p>
</div>

### Acerca del proyecto

Este proyecto tiene como objetivo desarrollar un modelo de pronóstico de la radiación solar para el corto plazo usando herramientas de aprendizaje profundo e información de las estaciones meteorológicas de la Comunidad de Castilla y León, España, que permitan la toma de decisiones alrededor de proyectos de generación solar fotovoltaica.

### Herramientas requeridas

* Python 3.11
* Entorno de ejecución de cuadernos Jupyter (JupyterLab, Google Colab o Kaggle Notebooks).

### Descripción del repositorio

Los archivos del repositorio se encuentran distribuidos principalmente en notebooks de Jupyter (`.ipynb`) y archivos de python (`.py`) que incluyen tanto los paquetes desarrollados, como cada una de las tareas de desarrollo y pruebas realizadas.

Adicionalmente se cuenta con directorios correspondientes a documentación, imágenes y diagramas de arquitectura o diseño, paquetes de python, y datos preparados almacenados de la siguiente manera:

* `docs`: Corresponde a la ruta donde se almacenan los datos de documentación sobre la monografía a ser entregada
* `imgs`: Directorio de imágenes y diagramas de arquitectura o de diseño del proyecto.
* `tools`: Directorio asociado a paquetes de python desarrollados para cumplir con los requerimientos correspondientes.
* `data`: Corresponde a la ruta donde se almacenan en formato comprimido algunos de los datos que han sido preparados previamente en el proceso de limpieza y estandarización de datos.
* `models`: Directorio de modelos generados para cada uno de los experimentos
* `models_cross`: Directorio asociado con los modelos generados por medio de validación cruzada.
* `config`: Directorio con cada uno de los archivos de configuración definidos para usarse con el paquete de entrenamiento de modelos `tools.model_generation.*`

### ¿Como se ejecuta el proceso?

* Para realizar la ejecución de los notebooks se recomienda realizar previamente la instalación de las librerías requeridas y usando `Python 3.11`.
* Para realizar la instalación de las librerías debemos:
  * Crear un entorno virtual de python por medio de la librería `virtualenv`.

    ```bash
    python -v venv .venv
    ```
  * Habilitamos el entorno virtual
    
    **Linux**
    ```bash
    source .venv/bin/activate
    ```
    **Windows**
    ```cmd
    .venv\Scripts\activate.bat
    ```
  * Instalamos las librerías requeridas del archivo `requirements.txt` incluido en el directorio principal del repositorio.
    ```bash
    pip install -r requirements.txt
    ```
* La ejecución se realiza siguiendo el orden numérico de los cuadernos Jupyter relacionados en el directorio principal del repositorio, partiendo del **Análisis Exploratorio** (`1_Analisis_Exploratorio.ipynb`) hasta la generación de **Métricas** (`9_Metricas.ipynb`).