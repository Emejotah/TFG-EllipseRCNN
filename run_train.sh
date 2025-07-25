#!/bin/bash

# Activar entorno virtual
source ~/.venv/bin/activate  

# Entrenar el modelo
python scripts/train.py --dataset=FDDB --accelerator=gpu
