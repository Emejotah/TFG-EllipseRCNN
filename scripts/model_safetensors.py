import torch
from safetensors.torch import save_file
from ellipse_rcnn.pl import EllipseRCNNModule

# Cargar el modelo desde el checkpoint .ckpt
ckpt_path = "C:/Users/mjgto/Desktop/Universidad/5Quinto/TFG_Infor/Proyecto/ellipse-rcnn/checkpoints/loss=0.02847-e=17.ckpt"
model = EllipseRCNNModule.load_from_checkpoint(ckpt_path)

# Obtener el state_dict (los pesos)
state_dict = model.state_dict()

# Guardar como .safetensors
save_file(state_dict, "model.safetensors")
