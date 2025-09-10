# TTA Configuration Evaluator

Este directorio contiene herramientas para evaluar y optimizar configuraciones de TTA (Test Time Augmentation) para mejorar la precisión del modelo Ellipse R-CNN.

## 🎯 Objetivo

El objetivo principal es encontrar la configuración óptima de TTA que:
- Minimice falsos positivos
- Reduzca objetivos perdidos (false negatives)
- Maximice la precisión y recall del modelo
- Reduzca la variabilidad en las métricas de evaluación

## 📁 Archivos

- `tta_configuration_evaluator.py`: Script principal de evaluación de configuraciones
- `tta_transforms.py`: Módulo con funciones de actualización de configuración
- `README.md`: Esta documentación

## 🔧 Configuraciones Evaluadas

El script evalúa diferentes combinaciones de parámetros:

### TTA_CONFIG
- `min_score_threshold`: Umbral mínimo de confianza para detecciones
- `consensuation_distance_threshold`: Distancia máxima para consensuar elipses

### QUALITY_CONFIG
- `high_quality_threshold`: Umbral para considerar detecciones de alta calidad
- `consistency_distance_base`: Base para calcular consistencia espacial
- `quality_exponent`: Exponente para scoring de calidad

### VALIDATION_CONFIG
- `center_deviation_threshold`: Desviación máxima permitida en centro (píxeles)
- `angle_deviation_threshold`: Desviación máxima permitida en ángulo (grados)
- `area_deviation_threshold`: Desviación máxima permitida en área (porcentaje)
- `center_weight`, `angle_weight`, `area_weight`: Pesos para calcular desviación total

## 🚀 Uso

### Instalación de dependencias

```bash
pip install torch torchvision matplotlib pandas numpy
```

### Ejecución básica

```bash
python tta_configuration_evaluator.py
```

### Opciones avanzadas

```bash
python tta_configuration_evaluator.py \
    --num_images 200 \
    --random_seed 123 \
    --data_path "../data/FDDB" \
    --device "cuda"
```

### Parámetros disponibles

- `--num_images`: Número de imágenes a evaluar (default: 100)
- `--random_seed`: Semilla aleatoria para reproducibilidad (default: 42)
- `--model_repo`: Repositorio del modelo en Hugging Face (default: "MJGT/ellipse-rcnn-FDDB")
- `--data_path`: Ruta al dataset FDDB (default: "../data/FDDB")
- `--device`: Dispositivo a usar (cuda/cpu, default: "cuda")

## 📊 Resultados

El script genera varios archivos de resultados:

### Archivos JSON
- `tta_config_evaluation_YYYYMMDD_HHMMSS.json`: Resultados detallados en formato JSON

### Archivos CSV
- `tta_config_summary_YYYYMMDD_HHMMSS.csv`: Resumen de métricas por configuración

### Visualizaciones
- `tta_config_evaluation_plots_YYYYMMDD_HHMMSS.png`: Gráficos comparativos

## 📈 Métricas Evaluadas

### Por cada configuración se calcula:

1. **Precision**: TP / (TP + FP)
2. **Recall**: TP / (TP + FN)
3. **False Positives**: Detecciones incorrectas
4. **False Negatives**: Objetivos perdidos
5. **Improvement Score**: Métrica compuesta que considera:
   - Reducción de falsos positivos vs modelo base
   - Reducción de objetivos perdidos vs modelo base
   - Mejora en F1-score

## 🏆 Configuraciones Predefinidas

### 1. Baseline
Configuración actual del sistema (valores por defecto)

### 2. Strict_Consensus
- Umbrales más estrictos para consensuar
- Mayor calidad requerida
- Menor tolerancia a desviaciones

### 3. Lenient_Consensus
- Umbrales más permisivos
- Acepta más variación en las predicciones
- Mayor inclusión de detecciones

### 4. No_Quality_Filter
- Desactiva filtros de calidad
- Incluye todas las detecciones por encima del umbral mínimo

### 5. Conservative_High_Quality
- Solo acepta detecciones de muy alta calidad
- Umbrales de desviación muy estrictos
- Enfoque conservador para minimizar falsos positivos

### 6. No_Validation
- Desactiva validación de consenso
- Umbrales de desviación muy altos (efectivamente deshabilitados)

### 7. Balanced_PR
- Configuración balanceada entre precisión y recall
- Pesos equilibrados para diferentes tipos de desviación

### 8. Area_Focused
- Enfatiza consistencia en el tamaño de las elipses
- Mayor peso para desviación de área
- Útil cuando el tamaño es crítico

## 🔍 Análisis de Resultados

### Interpretación del Improvement Score

- **Positivo**: La configuración TTA mejora respecto al modelo base
- **Negativo**: La configuración TTA empeora respecto al modelo base
- **Cercano a 0**: Mejoras marginales o sin cambios significativos

### Recomendaciones

1. **Para minimizar falsos positivos**: Usar configuraciones conservadoras (Conservative_High_Quality, Strict_Consensus)

2. **Para maximizar recall**: Usar configuraciones permisivas (Lenient_Consensus, No_Quality_Filter)

3. **Para balance óptimo**: Usar configuraciones balanceadas (Balanced_PR, Area_Focused)

4. **Para debugging**: Usar No_Validation para aislar efectos de consensuación

## 🛠️ Personalización

Para añadir nuevas configuraciones, modifica la función `define_configuration_grid()` en el script principal:

```python
configurations.append({
    'name': 'Mi_Configuracion_Custom',
    'tta_config': {
        'min_score_threshold': 0.75,
        # ... otros parámetros
    },
    'quality_config': {
        # ... parámetros de calidad
    },
    'validation_config': {
        # ... parámetros de validación
    }
})
```

## 📝 Logging

El script proporciona logging detallado:
- Información de configuración inicial
- Progreso de evaluación por configuración
- Métricas intermedias
- Resumen final con ranking de configuraciones

## ⚡ Tips de Optimización

1. **Número de imágenes**: Comenzar con 50-100 imágenes para pruebas rápidas, usar 200+ para evaluación final

2. **Random seed**: Usar la misma semilla para comparar configuraciones de forma justa

3. **Device**: Usar GPU (cuda) para evaluaciones más rápidas

4. **Memoria**: Si hay problemas de memoria, reducir el número de imágenes o usar CPU

## 🐛 Troubleshooting

### Error de importación
Verificar que el path esté correctamente configurado y que todos los módulos estén disponibles.

### Error de dataset
Verificar que la ruta al dataset FDDB sea correcta y que contenga las carpetas `images` y `labels`.

### Error de memoria GPU
Reducir el número de imágenes o cambiar a CPU con `--device cpu`.

## 📊 Ejemplo de Salida

```
🏆 Best Configuration: Balanced_PR
   Improvement Score: 0.245
   TTA Precision: 0.892
   TTA Recall: 0.867
   FP Reduction: 12
   FN Reduction: 8

📈 All Configurations (sorted by improvement):
 1. Balanced_PR          | Score:  0.245 | P: 0.892 | R: 0.867 | FP:  15 | FN:  18
 2. Strict_Consensus     | Score:  0.198 | P: 0.934 | R: 0.823 | FP:   8 | FN:  24
 3. Conservative_High_Q  | Score:  0.156 | P: 0.956 | R: 0.787 | FP:   5 | FN:  29
 ...
```

Esta evaluación ayuda a identificar sistemáticamente las mejores configuraciones para reducir la variabilidad y optimizar el rendimiento del modelo TTA.
