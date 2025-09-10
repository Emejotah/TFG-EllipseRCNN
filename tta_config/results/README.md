# TTA Configuration Evaluation Results

Este directorio contiene los resultados de las evaluaciones de configuraciones TTA.

## Archivos Generados

### JSON (Resultados Detallados)
- `tta_config_evaluation_YYYYMMDD_HHMMSS.json`: Resultados completos con todas las métricas por configuración

### CSV (Resumen Tabular)
- `tta_config_summary_YYYYMMDD_HHMMSS.csv`: Tabla resumen con métricas principales para análisis en Excel/LibreOffice

### PNG (Visualizaciones)
- `tta_config_evaluation_plots_YYYYMMDD_HHMMSS.png`: Gráficos comparativos de todas las configuraciones

## Estructura de Archivos

### JSON Content
```json
{
  "config_name": "Baseline",
  "config": {...},
  "base_precision": 0.85,
  "base_recall": 0.78,
  "tta_precision": 0.92,
  "tta_recall": 0.86,
  "base_fp_count": 12,
  "tta_fp_count": 8,
  "base_missed_count": 15,
  "tta_missed_count": 10,
  "improvement_score": 0.45
}
```

### CSV Columns
- Configuration: Nombre de la configuración
- Base_Precision: Precisión del modelo base
- Base_Recall: Recall del modelo base  
- TTA_Precision: Precisión con TTA
- TTA_Recall: Recall con TTA
- Base_FP: Falsos positivos del modelo base
- TTA_FP: Falsos positivos con TTA
- Base_FN: Falsos negativos del modelo base
- TTA_FN: Falsos negativos con TTA
- Improvement_Score: Puntuación de mejora compuesta

### PNG Plots
1. **Precision vs Recall**: Scatter plot comparando precisión y recall
2. **Configuration Improvement Scores**: Barras de puntuación de mejora (orden original)
3. **False Positives**: Conteo de falsos positivos por configuración
4. **Missed Targets**: Conteo de objetivos perdidos por configuración

## Nomenclatura de Timestamps

Formato: YYYYMMDD_HHMMSS
- YYYY: Año (4 dígitos)
- MM: Mes (2 dígitos)
- DD: Día (2 dígitos)
- HH: Hora (2 dígitos, 24h)
- MM: Minutos (2 dígitos)
- SS: Segundos (2 dígitos)

Ejemplo: `20250909_143022` = 9 de Septiembre 2025, 14:30:22
