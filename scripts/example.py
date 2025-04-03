#%%
import torch
from ellipse_rcnn.hf import EllipseRCNN  # This is the model with HF functionality included through PyTorchModelHubMixin
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ellipse_rcnn.utils.viz import plot_single_pred
import os


model = EllipseRCNN.from_pretrained("wdoppenberg/crater-rcnn")  # For the crater detection model
model.eval()

png = Image.open(r"docs\example_craters.png").convert("L")
img = to_tensor(png)
with torch.no_grad():
    pred = model([img])

plot_single_pred(img, pred)

