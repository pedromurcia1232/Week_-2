# Week12_12 Implementación del Mecanismo de Atención para Series de Tiempo y un Transformer Básico en Google Colab 
# Integrantes: 

DIEGO ALEJANDRO RUIZ ALFONSO - ppmurcia@ucundinamarca.edu.co 
PEDRO PASCUAL MURCIA VARGAS - ppmurcia@ucundinamarca.edu.co 
JHON EDUARD TINJACA CRUZ - jetinjaca@ucundinamarca.edu.co 
JULIAN DAVID SILVA GUZMAN - jdsilva@ucundinamarca.edu.co

# Librerías y Funciones Especiales Utilizadas

El proyecto se apoya en el ecosistema de **TensorFlow** y herramientas estándar de ciencia de datos:

### 1. Librerías Principales
*   **TensorFlow / Keras**: Para la construcción y entrenamiento de redes neuronales.
*   **NumPy**: Para manipulación de arreglos y generación de datos sintéticos.
*   **Pandas**: Para la carga de datos (CSV) y manejo de tablas de resultados.
*   **Matplotlib**: Para todas las visualizaciones y gráficas de series de tiempo.

### 2. Funciones y Capas Especiales
*   **`AttentionLayer` (Personalizada)**: Una clase que hereda de `tf.keras.layers.Layer` para implementar el mecanismo de atención manual usando pesos entrenables y activación `tanh`.
*   **`MultiHeadAttention`**: Capa de Keras que permite al modelo atender a múltiples subespacios de la secuencia en paralelo (corazón del Transformer).
*   **`LSTM`**: Capa recurrente con celdas de memoria para capturar dependencias a largo plazo.
*   **`LayerNormalization`**: Utilizada para estabilizar el entrenamiento en el Transformer.
*   **`GlobalAveragePooling1D`**: Para reducir la dimensión temporal de la secuencia a un vector fijo antes de la capa final.
*   **`MinMaxScaler` (Lógica manual)**: Implementada para normalizar la serie entre 0 y 1, asegurando que el MSE sea comparable entre modelos.


# 1. Preparación de Datos
Carga del Dataset: Se importa el conjunto de datos histórico de pasajeros de líneas aéreas internacionales. Es una serie de tiempo clásica que muestra tendencia y estacionalidad.
Normalización: Se aplica el escalado Min-Max para transformar los valores al rango [0, 1]. Esto es crucial para que las funciones de activación de las redes neuronales (como la tanh o sigmoid) funcionen de manera óptima.
Ventanas Deslizantes (Sliding Windows): Se convierte la serie en un problema de aprendizaje supervisado. Con una ventana de 40 (WINDOW_SIZE), el modelo mira los 40 meses anteriores para predecir el mes 41.

# 2. Arquitecturas de los Modelos
Mecanismo de Atención Personalizado:
AttentionLayer: Calcula un peso (importancia) para cada uno de los 40 pasos de tiempo.
softmax: Asegura que todos los pesos sumen 1.
reduce_sum: Combina la información ponderada en un único 'vector de contexto' que resume lo más importante de la secuencia.
Transformer:
Utiliza MultiHeadAttention, lo que permite al modelo prestar atención a diferentes patrones (como la tendencia a largo plazo y la estacionalidad a corto plazo) simultáneamente.
Incluye LayerNormalization y conexiones residuales para dar estabilidad al entrenamiento.
LSTM (Long Short-Term Memory):
Es nuestra línea base. Está diseñada específicamente para recordar información a largo plazo y procesar datos secuencialmente paso a paso.

# 3. Resultados y Análisis (MSE)
LSTM (V:40) - MSE 0.0275: Fue el mejor modelo. Al procesar paso a paso, tiene un 'sesgo inductivo' que le ayuda a entender el orden temporal incluso con pocos datos.
Atención (V:40) - MSE 0.0299: Muy cercano al LSTM, demostrando que identificar qué meses pasados son importantes es casi tan efectivo como la memoria recurrente.
Experimento Ventana 60 - MSE 0.0316: Al aumentar la ventana a 60, el error subió. Esto sucede porque, aunque el modelo ve más pasado, tenemos menos ejemplos totales para entrenar (ya que 'consumimos' más datos para formar la primera ventana).
Transformer (V:40) - MSE 0.0470: Tuvo el error más alto. Los Transformers son muy potentes pero 'hambrientos de datos'. Con solo 144 registros, el modelo tiende a sobreajustar o no logra aprender las relaciones complejas de sus múltiples cabezales de atención.

# 4. Visualización Final
Las gráficas comparativas muestran que el LSTM sigue mucho más de cerca las curvas reales del set de prueba, mientras que el Transformer produce una predicción más suavizada pero menos precisa para este volumen de datos.

# Conclusiones y Resultados Finales

Tras completar el desarrollo y los experimentos con la serie de tiempo **Air Passengers**, se extraen las siguientes conclusiones:

### 1. Comparativa de Rendimiento (MSE)
*   **LSTM (V:40)**: Fue el modelo con mejor desempeño (~0.0275). Su arquitectura recurrente es ideal para series cortas con patrones claros.
*   **Atención (V:40)**: Mostró resultados muy competitivos (~0.0299), probando que ponderar pasos temporales clave es una estrategia robusta.
*   **Transformer (V:40)**: Tuvo el error más alto (~0.0470). Debido al tamaño limitado del dataset (144 registros), el Transformer no pudo aprovechar su capacidad de atención multi-cabezal sin sobreajustar.

### 2. Impacto de la Ventana Temporal (WINDOW_SIZE)
*   Aumentar la ventana de **40 a 60** pasos resultó en una degradación del rendimiento (MSE aumentó a ~0.0316).
*   **Lección aprendida**: Ventanas más grandes no siempre son mejores. En datasets pequeños, una ventana más grande reduce la cantidad de ejemplos de entrenamiento disponibles, lo que puede perjudicar la capacidad de aprendizaje del modelo.

### 3. Consideraciones Técnicas
*   **Sesgo Inductivo**: Las RNNs (como LSTM) tienen un sesgo hacia la secuencialidad que las hace más eficientes en problemas con pocos datos comparado con los Transformers, que son arquitecturas 'data-hungry' (hambrientas de datos).
*   **Normalización**: El uso de escalado Min-Max fue fundamental para estabilizar el entrenamiento de todas las arquitecturas.
