# Simulación pipeline Preprocesamiento y Predicción

Este proyecto implementa un pipeline simulado 
para predecir la **probabilidad de fracaso** tras la reducción de sedación en pacientes intubados. El modelo se basa en variables tabulares de la **ficha médica del paciente**, procesadas mediante un preprocesamiento reproducible y alimentadas a un clasificador `CatBoost`.

> ⚠️ **Nota:** Este modelo **no** es de los modelos preparados y evaluados para el proyecto. Ha sido específicamente entrenado con 10 variables de la ficha médica del paciente para demostrar el flujo de predicción en un entorno controlado.

## Instalación y entorno
Las dependencias del proyecto se encuentran en `requirements.txt` se recomienda instalar en ambiente virtual. 

## Uso
El script principal es `main.py`, que debe ejecutarse con los siguientes argumentos:

- `--input`: Ruta al archivo `.json` que contiene los datos crudos de la ficha medica de pacientes. **(Obligatorio)**
- `--output`: Ruta donde se guardará el archivo de predicciones `.csv`. (Opcional, por defecto: `predicciones.csv`)
- `--bundle`: Ruta al archivo `.pkl` que contiene el modelo `CatBoost` entrenado y su respectivo preprocesador. (Opcional, por defecto: `models/catboost_pipeline.pkl`)

###  Ejemplo de ejecución

```bash
python main.py \
  --input data/pacientes_nuevos.json \
  --bundle models/catboost_pipeline.pkl \
  --output resultados.csv
```

Esto genera un archivo `resultados.csv` con tres columnas:
- `_id`: identificador de la ficha del paciente
- `prob_fracaso`: probabilidad estimada de fracaso en extubación tras bajar la sedación del paciente.
- `pred_clase`: Se usa umbral estándar de 0.5 para hacer punto de corte. Por debajo de 0.5 la probabilidad indica éxito, por encima o igual indica fracaso. 

## Formato de datos de entrada

El archivo de entrada debe estar en formato `.json` y contener una **lista de pacientes**, donde cada uno es un diccionario con los campos de su ficha médica. El modelo extraerá automáticamente las variables necesarias para la predicción. 

Se puede pasar directamente los datos como vienen en el endpoint `export-medical-record` de la API retexo, pero a continuación se listan las variables efectivamente usadas en el flujo simulado. 

### Campos que **sí** utiliza el modelo

| Campo | Tipo esperado | Ejemplo (dentro del JSON) | Notas |
|-------|---------------|---------------------------|-------|
| `_id` | `string` | `"65e31aaaf3883313f65d6924"` | Identificador único del registro. |
| `apache2Ingreso` | `string` o `number` | `"18"` | Se discretiza en 9 categorías de riesgo. |
| `funcionalidad` | **lista** con un diccionario `{ "value": int, "name": str }` | `"funcionalidad": [{ "value": 5, "name":"100: Independencia"}]` | Se toma **solo** el primer `value`. |
| `fragilidad` | lista `[ { "value": int, "name": str } ]` | `"fragilidad":[{ "value":2, "name":"Pre-fragilidad" }]` | Igual que arriba. |
| `AntecedentesConsumoDrogas` | lista `[ { "value": int, "name": str } ]` | `"AntecedentesConsumoDrogas":[{ "value":1, "name":"No" }]` | Se usa `value` (0 = No, 1 = Recreativo, 2 = Abuso). |
| `antecedenteQx` | **cualquier** valor / lista | `"antecedenteQx":[{ "value":0, "name":"-- Sin Antecedente Quirúrgico --"}]` | Se interpreta a binario (1 = tiene antecedente). |
| `antecedentesAbusoAlcohol` | lista `[ { "value": int, "name": str } ]` | `"antecedentesAbusoAlcohol":[{ "value":2, "name":"No" }]` | 0 = No, 1 = Sí. |
| `antecedentesHabitoTabaquico` | lista `[ { "value": int, "name": str } ]` | `"antecedentesHabitoTabaquico":[{ "value":2, "name":"No" }]` | 0 = No, 1 = Sí. |
| `imc` | `number` | `25.0` | Se discretiza: 0 =Bajo, 1 = Normal, 2 = Sobrepeso/Obesidad. |
| Fechas `fechaIngresoHospital`, `fechaIngresoAUci`, `fechaIntubacion` | `string` `dd/mm/yyyy hh:mm` | `"fechaIntubacion":"02/03/2024 09:22"` | Usadas para calcular días hasta intubación. |

> Todos los demás campos del JSON **se ignoran** durante la predicción, por lo que pueden conservarse sin problema.

---

### Ejemplo mínimo válido

```json
[
  {
    "_id": "65e31aaaf3883313f65d6924",
    "apache2Ingreso": "18",
    "funcionalidad": [{ "value": 5, "name": "100: Independencia" }],
    "fragilidad": [{ "value": 2, "name": "Pre-fragilidad" }],
    "AntecedentesConsumoDrogas": [{ "value": 1, "name": "No" }],
    "antecedenteQx": [{ "value": 0, "name": "-- Sin Antecedente Quirúrgico --" }],
    "antecedentesAbusoAlcohol": [{ "value": 2, "name": "No" }],
    "antecedentesHabitoTabaquico": [{ "value": 2, "name": "No" }],
    "imc": 25.0,
    "fechaIngresoHospital": "27/02/2024 09:21",
    "fechaIngresoAUci": "01/03/2024 10:22",
    "fechaIntubacion": "02/03/2024 09:22"
  }
]
```

## Flujo del pipeline de predicción

  
El flujo general sigue los siguientes pasos:


### 1. Carga de datos

Archivo de entrada: `JSON` con registros de pacientes.

- Se carga utilizando `json.load()`.
- El contenido debe ser una **lista de diccionarios**, donde cada uno representa a un paciente.


---

### 2. Preprocesamiento

Archivo: `pipeline/preprocessor.py`

#### a. Selección y limpieza de variables

- Se filtran únicamente las **variables relevantes** para el modelo (ver sección anterior).
- Se extrae el campo `.value` en los campos tipo lista (ej. `funcionalidad`, `fragilidad`, etc).
- Se maneja la lógica para categorizar IMC, fragilidad y otras variables discretizadas.

#### b. Cálculo de variable derivada: `días hasta intubación`

- Se parsean las fechas `fechaIngresoHospital`, `fechaIngresoAUci`, y `fechaIntubacion`.
- Se calcula el número de días entre hospitalización e intubación, y días entre ingreso UCI a intubación. 

---

### 3. Transformación a `DataFrame`

Una vez extraídas las variables, se construye un `DataFrame` de Pandas con las 10 variables procesadas.

- Se construye un `DataFrame` de Pandas con las variables procesadas.
- Se asegura la existencia de los campos requeridos por el modelo.
- **Los valores faltantes (`NaN`) son imputados** automáticamente con los mismos valores usados durante el entrenamiento (ej. medianas, constantes, etc.).

---

### 5. Predicción

- Se aplica el método `.predict_proba()` sobre el `DataFrame`.
- Se obtiene la **probabilidad de clase 1** (fracaso), la cual es la que se reporta.
- También se obtiene la clase final (`0` o `1`) según el umbral por defecto (`0.5`).

El resultado es una lista de predicciones por paciente, que incluye:
```json
[
  {
    "_id": "65e31aaaf3883313f65d6924",
    "prob_fracaso": 0.348,
    "pred_clase": 0
  ...
]
