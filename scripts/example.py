#%%
import torch
from ellipse_rcnn.hf import EllipseRCNN  # This is the model with HF functionality included through PyTorchModelHubMixin
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ellipse_rcnn.utils.viz import plot_single_pred
import os

#%%

model = EllipseRCNN.from_pretrained("wdoppenberg/crater-rcnn")  # For the crater detection model
model.eval()

png = Image.open(r"..\docs\example_craters.png").convert("L")
img = to_tensor(png)
with torch.no_grad():
    pred = model([img])

plot_single_pred(img, pred)

# %%
print("Directorio actual:", os.getcwd())

# %%

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

# %%
import torch
import pytorch_lightning as pl
from pytorch_lightning.accelerators.cuda import CUDAAccelerator

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Lightning version:", pl.__version__)
print("CUDAAccelerator available in Lightning:", CUDAAccelerator.is_available())


# %% EJEMPLO CON FDDB
from ellipse_rcnn.pl import EllipseRCNNModule
from safetensors.torch import load_file

model = EllipseRCNNModule()
model.load_state_dict(load_file(r"C:\Users\mjgto\Desktop\Universidad\5Quinto\TFG_Infor\Proyecto\ellipse-rcnn-FDDB\model.safetensors"))
model.eval()

png = Image.open(r"..\docs\sims_face.jpg").convert("L")
img = to_tensor(png)
with torch.no_grad():
    pred = model([img])

plot_single_pred(img, pred)


# %%
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=r"C:\Users\mjgto\Desktop\Universidad\5Quinto\TFG_Infor\Proyecto\ellipse-rcnn-FDDB\model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="MJGT/ellipse-rcnn-FDDB",
)
# %%
