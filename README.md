# Práctica Calificada 4: Proyecto 3 - Chunking y Construcción de un Vector Store

Este proyecto implementa, analiza y compara cuantitativa y cualitativamente tres estrategias de segmentación de texto (*chunking*) para optimizar la recuperación en sistemas de Generación Aumentada por Recuperación (RAG).

## Objetivo

El objetivo principal es evaluar el impacto de la granularidad y la coherencia semántica en la recuperación de información. Para ello, se segmenta un corpus largo autodetenido sobre la película ***Interstellar*** (de aproximadamente 5,600 caracteres) mediante diferentes técnicas, se indexan los fragmentos resultantes en un almacén de vectores local y se evalúa la efectividad del retrieval frente a un benchmark de consultas predefinidas usando las métricas de **Recall@k** y **MRR@k** (Mean Reciprocal Rank).

---

## Cuaderno Base

El desarrollo se fundamenta en el material del curso:
* **Base:** Cuaderno21-CC0C2.ipynb - Búsqueda Semántica y RAG local.
* **Evaluación:** Cuaderno23-CC0C2.ipynb - Métricas de evaluación de recuperación.

---

## Resumen de la Línea Base (Estrategia A)

* **Estrategia:** *Fixed-Size Chunking* (Segmentación por caracteres de tamaño fijo).
* **Parámetros:** `chunk_size = 400`, `chunk_overlap = 50` utilizando `CharacterTextSplitter`.
* **Características:** Realiza divisiones lineales basándose únicamente en la longitud de caracteres. No tiene en cuenta la semántica ni la estructura natural de los párrafos del documento, provocando cortes abruptos.

---

## Modificación Realizada (Estrategias B y C)

Para superar las limitaciones de la línea base, se implementaron e integraron dos variantes de segmentación:

1. **Estrategia B: Recursive Character Chunking (Variante)**
   * **Herramienta:** `RecursiveCharacterTextSplitter`.
   * **Parámetros:** `chunk_size = 400`, `chunk_overlap = 50`, `separators = ["\n\n", "\n", " ", ""]`.
   * **Características:** Intenta mantener juntos los párrafos y oraciones lógicas dividiendo de manera jerárquica, evitando fragmentaciones de frases completas.

2. **Estrategia C: Semantic Chunking (Variante Avanzada)**
   * **Implementación:** Algoritmo en Python nativo que segmenta el texto basándose en la similitud semántica.
   * **Flujo:**
     1. Divide el texto en oraciones individuales.
     2. Genera embeddings para cada oración con el modelo `BAAI/bge-small-en-v1.5`.
     3. Calcula la distancia semántica (`1 - SimilitudCoseno`) entre oraciones consecutivas.
     4. Corta el texto en los puntos de transición donde la distancia supera el **percentil 85** (umbral adaptativo).
   * **Características:** Los límites del chunk son flexibles y representan divisiones naturales de temas en el documento.

---

## 5. Cómo Ejecutar el Notebook

Siga los siguientes pasos para reproducir las celdas:

### Requisitos Previos
Asegúrese de contar con Python 3.8+ y las siguientes librerías instaladas:
```bash
pip install torch transformers pandas numpy
```

### Ejecución
1. Abra una terminal en el directorio del proyecto 
2. Inicie el servidor de Jupyter:
   ```bash
   jupyter notebook
   ```
3. Abra el archivo [PC4.ipynb].
4. Ejecute todas las celdas secuencialmente (`Cell > Run All`). La ejecución utilizará la GPU si está disponible; de lo contrario, se ejecutará en CPU de forma determinista gracias a la semilla 42 fija.
---

## Principales Resultados

A continuación se detallan las métricas de recuperación evaluadas sobre un benchmark de 5 consultas específicas sobre *Interstellar* (evaluando a un top $k=3$ y $k=5$):

| Estrategia | Chunks Creados | Tamaño Promedio (chars) | Recall@3 | MRR@3 | Recall@5 | MRR@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Size Chunking (A)** | 18 | 348.44 | 0.80 | 0.50 | 1.00 | 0.54 |
| **Recursive Character (B)** | 26 | 253.81 | 0.80 | 0.50 | 1.00 | 0.55 |
| **Semantic Chunking (C)** | 7 | 897.71 | 1.00 | 0.90 | 1.00 | 0.90 |

### Análisis Cualitativo
* **Fixed-Size:** Realizó cortes lineales por caracteres, fragmentando algunas explicaciones conceptuales en la mitad (por ejemplo, dividiendo los datos del planeta de Miller entre dos chunks). Logró un Recall@3 de 0.80 y MRR@3 de 0.50.
* **Recursive Character:** Aunque respeta los límites lógicos, el tamaño promedio pequeño de los chunks (~254 caracteres) limitó levemente la densidad semántica de la respuesta completa dentro del top-3, igualando el Recall@3 de 0.80 y obteniendo un MRR@3 de 0.50.
* **Semantic:** Agrupó oraciones coherentemente por subtemas (Gargantua, Miller, el Teseracto). Logró un Recall@3 perfecto de 1.00 y un MRR@3 de 0.90, demostrando ser la estrategia más efectiva para mantener el contexto unificado y entregar respuestas completas.

---

## Limitaciones

1. **Sensibilidad del Umbral Semántico:** El número de chunks semánticos depende fuertemente del percentil de corte seleccionado. Un percentil muy bajo sobre-segmentará el texto, mientras que uno muy alto creará macro-chunks demasiado largos.
2. **Costo Computacional de Embeddings:** El Semantic Chunking requiere la generación previa de embeddings para todas las oraciones del corpus de forma secuencial, elevando el costo computacional.
3. **Métrica de Similitud Simple:** Se empleó producto punto de vectores normalizados (similitud coseno) sin mecanismos de refinamiento como Reranking o RAG híbrido.

---

## Qué se muestra en el Video Técnico

El video correspondiente a la defensa asíncrona presenta:
1. **Introducción:** Explicación del problema del chunking y su relevancia en RAG.
2. **Explicación del Código:** Recorrido por el notebook y justificación matemática de la similitud coseno con embeddings L2 normalizados.
3. **Codificación en Vivo - Ejercicio A (Cambio Común):** Modificación del tamaño de chunk a 600 caracteres y visualización en vivo de cómo impacta la cantidad de fragmentos y el Recall@3.
4. **Codificación en Vivo - Ejercicio B (Cambio de Proyecto):** Ajuste del percentil de corte del *Semantic Chunker* (de 85 a 75) para observar el incremento en la cantidad de chunks y la regeneración dinámica del gráfico de distancias.
5. **Respuestas a las Preguntas Avanzadas:** Defensa oral de las preguntas detalladas en el notebook.

---

## Declaración de Autoría y Uso de IA

```text
Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta Práctica.
Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.
```

---

## Puente al Curso

Este proyecto conecta con:
* **Chunking y Vectorización:** Vínculo estrecho entre la estructura del texto y su representación en espacios métricos continuos.
* **Evaluación de Retrieval:** Uso empírico y cálculo de métricas de ranking (`Recall`, `MRR`) para la optimización cuantitativa de sistemas lingüísticos aplicados.
