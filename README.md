# 🌿 AgroDetect

**Sistema Inteligente para la Detección de Enfermedades y Plagas Foliares en el Cultivo de Café**

> Proyecto final de Inteligencia Artificial aplicado a la caficultura de Comayagua, Honduras.

[![Estado](https://img.shields.io/badge/estado-finalizado-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)]()


---
- **Url Streamlit ** —  ([Streamlit](https://agrodetect-grup4.streamlit.app/) )
## 📖 Índice

- [Descripción del proyecto](#-descripción-del-proyecto)
- [Problema que resuelve](#-problema-que-resuelve)
- [Objetivos](#-objetivos)
- [Enfermedades y plagas detectadas](#-enfermedades-y-plagas-detectadas)
- [Arquitectura de la solución](#️-arquitectura-de-la-solución)
- [Resultados del modelo](#-resultados-del-modelo)
- [Tecnologías utilizadas](#️-tecnologías-utilizadas)
- [Datasets utilizados](#-datasets-utilizados)
- [Requisitos previos](#-requisitos-previos)
- [Instrucciones para ejecutar la aplicación](#️-instrucciones-para-ejecutar-la-aplicación)
- [Reentrenar el modelo (opcional)](#-reentrenar-el-modelo-opcional)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Beneficiarios](#-beneficiarios)
- [Escalabilidad](#-escalabilidad)
- [Mantenimiento y mejoras futuras](#-mantenimiento-y-mejoras-futuras)
- [Equipo](#-equipo)
- [Licencia](#-licencia)
- [Referencias](#-referencias)

---

## 🌱 Descripción del proyecto

El café es uno de los pilares de la economía hondureña, generando ingresos directos e indirectos para cientos de miles de familias. Sin embargo, la caficultura enfrenta amenazas recurrentes como la **roya del café** (*Hemileia vastatrix*), la **cercospora**, la **phoma**, y plagas como la **araña roja** y el **minador de la hoja**, que reducen el rendimiento y la calidad del grano.

**AgroDetect** es un sistema basado en **visión artificial** y **aprendizaje profundo** que, a partir de una fotografía tomada con un teléfono móvil, clasifica el estado de salud de una hoja de café y sugiere la enfermedad o plaga presente, funcionando como una primera línea de apoyo al diagnóstico agronómico para productores con acceso limitado a asistencia técnica especializada.

Este repositorio contiene la **versión final** del proyecto: el modelo ya entrenado y evaluado, la aplicación web lista para ejecutarse, y el notebook completo con todo el proceso de desarrollo (comparación de arquitecturas, balanceo del dataset, interpretabilidad con Grad-CAM, etc.).

## 🎯 Problema que resuelve

En la actualidad, el diagnóstico de enfermedades foliares depende casi exclusivamente de la inspección visual del productor o de un técnico agrónomo, un método lento, subjetivo y poco accesible en fincas pequeñas, medianas o de difícil acceso geográfico —una realidad común en la región de Comayagua, Honduras.

**AgroDetect** busca reducir esa brecha ofreciendo un diagnóstico **rápido, objetivo y accesible desde un smartphone**.

## 🎯 Objetivos

**Objetivo general:** desarrollar y desplegar un sistema de IA para la detección y clasificación de enfermedades y plagas foliares del café, como herramienta de apoyo al diagnóstico fitosanitario de productores de Comayagua, Honduras.

**Objetivos específicos:**

- Recopilar y curar un dataset de imágenes de hojas de café sanas y afectadas.
- Entrenar y **comparar formalmente distintas arquitecturas CNN** (MobileNetV2, EfficientNetB0, ResNet50) mediante *transfer learning*, seleccionando la más adecuada según su desempeño y su viabilidad de despliegue.
- Evaluar el modelo con métricas estándar (exactitud, precisión, sensibilidad, F1-score) y técnicas de interpretabilidad (**Grad-CAM**).
- Desarrollar una interfaz web accesible para subir/capturar fotos y recibir diagnóstico.
- Integrar un módulo de recomendaciones agronómicas basado en un LLM, consumido mediante API key.
- Desplegar la solución en una plataforma en la nube gratuita.
- Definir una estrategia de mantenimiento y mejora continua.

## 🍃 Enfermedades y plagas detectadas

| Clase | Tipo | Alcance |
|---|---|---|
| Roya (*Hemileia vastatrix*) | Enfermedad | Clasificación de la enfermedad |
| Cercospora | Enfermedad | Presencia/ausencia en la hoja |
| Phoma | Enfermedad | Presencia/ausencia en la hoja |
| Araña roja (*red spider mite*) | Plaga | Presencia/ausencia en la hoja |
| Minador de la hoja (*leaf miner*) | Plaga | Presencia/ausencia en la hoja |

*(+ clase adicional: hoja sana)*

## 🏗️ Arquitectura de la solución

```
📷 Foto de la hoja
      │
      ▼
🖥️ Interfaz Streamlit (usuario sube/captura la imagen)
      │
      ▼
🧠 Modelo CNN (MobileNetV2, entrenado con Transfer Learning)
   Cargado directamente desde el archivo versionado en este repositorio
   (no requiere un servicio de inferencia independiente)
      │
      ▼
🔎 Clasificación (sana / roya / cercospora / phoma / araña roja / minador)
   + nivel de confianza, contrastado contra un umbral de decisión
      │
      ▼
💬 API de Groq (LLM) ──► Recomendaciones agronómicas, mediante API key
      │
      ▼
🖥️ Interfaz Streamlit ──► Resultado + recomendación al productor
```

Todo el flujo corre en un **único servicio de Streamlit**: la app carga el modelo de visión ya entrenado (archivo `.keras` incluido en este repositorio), clasifica la imagen, y llama directamente a la **API de Groq** con una API key (gestionada como *secret*, nunca en el código) para generar las recomendaciones de manejo agronómico — sin necesidad de un backend independiente.

## 📊 Resultados del modelo

Se compararon formalmente **MobileNetV2** y **EfficientNetB0** bajo el mismo protocolo de entrenamiento (extracción de características + *fine-tuning*), seleccionando la arquitectura ganadora combinando F1-score y tamaño del modelo (relevante para el despliegue gratuito). **MobileNetV2** resultó ganadora (F1 = 0.882 vs. 0.826 de EfficientNetB0, y solo 2.42M de parámetros).

Tras ampliar el dataset (nuevas fuentes de roya combinadas + imágenes en primer plano de la lesión para las clases más débiles) y una ronda de ajuste fino enfocada, el modelo final alcanzó:

| Métrica | Valor |
|---|---|
| Exactitud (test) | 0.896 |
| F1-score macro (test) | 0.888 |

Se utilizó **Grad-CAM** para verificar que el modelo aprende patrones visuales relevantes (la hoja y sus lesiones) y no artefactos del fondo, y para diagnosticar las causas de confusión entre clases visualmente similares. Los detalles completos (matriz de confusión, reporte por clase, mapas de calor) están documentados en [`docs/informe_final.pdf`](./docs/informe_final.pdf).

## 🛠️ Tecnologías utilizadas

| Categoría | Herramienta / Tecnología |
|---|---|
| Lenguaje de programación | Python |
| Framework de Deep Learning | TensorFlow / Keras |
| Procesamiento de imágenes | OpenCV, Pillow |
| Análisis numérico y de datos | NumPy, pandas |
| Métricas y evaluación | scikit-learn |
| Visualización / interpretabilidad | Matplotlib, Grad-CAM |
| Entrenamiento y experimentación | Google Colab (GPU gratuita) |
| Interfaz de usuario | Streamlit |
| Alojamiento del modelo entrenado | GitHub (este repositorio) |
| Recomendaciones mediante LLM | API de Groq (API key) |
| Control de versiones | Git y GitHub |
| Documentación y análisis | Jupyter Notebook |

## 📊 Datasets utilizados

El proyecto trabaja con imágenes de repositorios públicos ya etiquetados:

- **RoCoLe** — A Robusta Coffee Leaf Images Dataset ([Mendeley Data](https://data.mendeley.com/datasets/c5yvn32dzg/2) / [Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset))
- **CLR-Dataset** ([GitHub](https://github.com/dvelaren/clr-dataset)) — varias versiones combinadas para la clase roya, a fin de aumentar su diversidad
- **Coffee Leaf Disease** y variantes ([Roboflow Universe](https://universe.roboflow.com/dame-23kub/coffee-leaf-disease-kc1ca))
- **Ethiopian Coffee Leaf Disease** ([Kaggle](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease))
- **JMuBEN Coffee Dataset** ([Kaggle](https://www.kaggle.com/datasets/noamaanabdulazeem/jmuben-coffee-dataset))
- **Red Spider Mite** y **ROCC** ([Roboflow Universe](https://universe.roboflow.com/lance-eugene/red-spider-mite))
- Conjunto adicional de imágenes en primer plano de la lesión, incorporado para mejorar el reconocimiento de cercospora y araña roja

> La lista completa de repositorios y referencias bibliográficas está disponible en [`docs/informe_final.pdf`](./docs/informe_final.pdf).

## ✅ Requisitos previos

- **Python 3.10 o superior**
- **pip** (gestor de paquetes de Python)
- Una **API key de Groq**, gratuita, obtenida en [console.groq.com](https://console.groq.com) (necesaria únicamente para el módulo de recomendaciones)
- Git (para clonar el repositorio)

No se requiere GPU ni ninguna cuenta adicional para *ejecutar* la aplicación: el modelo ya viene entrenado y listo dentro del repositorio.

## ▶️ Instrucciones para ejecutar la aplicación

**1. Clonar el repositorio**

```bash
git clone https://github.com/Izamar-0302/AgroDetect.git
cd AgroDetect
```

**2. Crear y activar un entorno virtual** (recomendado)

```bash
python -m venv venv

# Activar en Linux/Mac:
source venv/bin/activate

# Activar en Windows:
venv\Scripts\activate
```

**3. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

**4. Configurar la API key de Groq**

Crea el archivo `.streamlit/secrets.toml` (si no existe) con el siguiente contenido:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "tu_api_key_aqui"
```

> ⚠️ Nunca subas este archivo a GitHub. Ya está incluido en `.gitignore`. Si vas a desplegar en Streamlit Community Cloud, configura esta misma variable en **App settings → Secrets** dentro de la plataforma, en vez de usar el archivo local.

**5. Ejecutar la aplicación**

```bash
streamlit run app.py
```

**6. Abrir en el navegador**

Streamlit abrirá automáticamente una pestaña en `http://localhost:8501`. Si no se abre sola, copia esa URL en tu navegador.

**7. Usar la aplicación**

1. Sube o captura una fotografía de una hoja de café.
2. El modelo de visión (cargado desde `model/` en este mismo repositorio) clasifica la imagen.
3. Si la confianza supera el umbral definido, la app consulta la API de Groq y muestra recomendaciones de manejo agronómico.
4. Consulta el resultado y, de ser necesario, contacta a un técnico del IHCAFE.

## 🔁 Reentrenar el modelo (opcional)

No es necesario para usar la aplicación, pero si quieres reentrenar o experimentar con el modelo:

1. Abre [`notebooks/AgroDetect_Entrenamiento.ipynb`](./notebooks/) en Google Colab (se recomienda activar GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*).
2. Ejecuta las celdas en orden: descarga y organización del dataset, comparación de arquitecturas, evaluación, diagnóstico con Grad-CAM, y exportación del modelo.
3. Al finalizar, descarga el archivo `.keras`/`.h5` generado y el `config_app.json` desde `/content/modelo_final/`, y reemplaza los archivos correspondientes en la carpeta `model/` de este repositorio.

## 📁 Estructura del repositorio

```
agrodetect/
├── .streamlit/
│   └── secrets.toml                     # Configuración de secretos / credenciales locales
├── Dataset/                             # Organización de datasets
├── docs/
│   └── anteproyecto_AgroDetect_overl... # Documentación y borrador del anteproyecto
├── model/
│   ├── agrodetect_mobilenetv2.keras    # Modelo entrenado (MobileNetV2)
│   └── config_app.json                 # Configuración de clases / parámetros de la app
├── notebooks/
│   └── AgroDetect_Entrenamiento.ipynb   # Notebook de entrenamiento (Google Colab / Jupyter)
├── .gitattributes                       # Configuración de Git (LFS, finales de línea, etc.)
├── README.md                            # Descripción y guía general del proyecto
├── app.py                               # Aplicación principal de Streamlit
└── requirements.txt                     # Dependencias del entorno de Python
```

## 👥 Beneficiarios

- Pequeños y medianos productores de café de Comayagua.
- Técnicos agrónomos y extensionistas del IHCAFE.
- Cooperativas y asociaciones de caficultores.
- Estudiantes y docentes de carreras agronómicas.

## 📈 Escalabilidad

Al estar el modelo desacoplado de la interfaz (se carga desde un archivo versionado, no está incrustado en la lógica de la app), puede actualizarse o reemplazarse por una arquitectura distinta sin modificar el código de Streamlit. Si el número de usuarios creciera lo suficiente, el modelo podría migrarse a un servicio de inferencia independiente sin rediseñar la aplicación.

## 🔧 Mantenimiento y mejoras futuras

- Reentrenamiento periódico del modelo con nuevas imágenes (incluyendo fotos aportadas por los propios usuarios, con su consentimiento).
- Versionado de los modelos entrenados dentro de este repositorio.
- Actualización del *prompt* del módulo de recomendaciones.
- Futuras mejoras: incorporación de la broca del café, detección de coinfecciones (más de una enfermedad presente en la misma hoja), app móvil offline, mapa de incidencia por zona geográfica.

## 👨‍💻 Equipo

Proyecto desarrollado para la asignatura de **Inteligencia Artificial**, UNAH — Campus Comayagua (UNAH-CURC).

- Jeimy Jazmín Palma Santos — 20221900269
- Ángeles Izamar Euceda Herrera — 20221930061
- Kilver Said Nolasco Parada — 20221930129

**Docente:** Asalia Alejandra Zavala

## 📄 Licencia

Este proyecto se distribuye bajo la licencia MIT. Ver [LICENSE](./LICENSE) para más detalles.

## 📚 Referencias

Las referencias completas de los datasets y la bibliografía utilizada están disponibles en la sección de **Referencias bibliográficas** del [informe final](./docs/informe_final.pdf).

---

<p align="center">Hecho con ☕ y 🧠 en Comayagua, Honduras</p>
