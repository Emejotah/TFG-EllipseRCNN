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


# %% EJEMPLO CON FDDB - Test Time Augmentation
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ellipse_rcnn.utils.viz import plot_single_pred
import torchvision.transforms as T
import os
from ellipse_rcnn.pl import EllipseRCNNModule
from safetensors.torch import load_file
from ellipse_rcnn.tta.tta_transforms import tta_predict

# Load the trained model
model = EllipseRCNNModule()
model.load_state_dict(load_file(r"C:\Users\mjgto\Desktop\Universidad\5Quinto\TFG_Infor\Proyecto\ellipse-rcnn-FDDB\model.safetensors"))
model.eval()
device = torch.device("cpu")
model.to(device)

# Load and preprocess the image
image_path = r"..\docs\sims_face.jpg"
if not os.path.exists(image_path):
    print(f"Error: Image not found at '{image_path}'")
else:
    # Load image
    png = Image.open(image_path).convert("L")
    img_tensor = to_tensor(png)
    
    print(f"Image shape: {img_tensor.shape}")
    
    # Standard prediction (no TTA)
    print("\n=== Standard Prediction (No TTA) ===")
    with torch.no_grad():
        standard_pred = model([img_tensor])
    
    print(f"Standard prediction: {len(standard_pred[0]['ellipse_params'])} ellipses detected")
    plot_single_pred(img_tensor, standard_pred)
    
    # TTA Prediction without consensuation
    print("\n=== TTA Prediction (Individual Augmentations) ===")
    tta_pred_individual = tta_predict(
        model=model,
        image_tensor=img_tensor, 
        device=device,
        min_score=0.75,
        consensuate=False,  # Don't consensuate - show all individual predictions
        visualize=True      # Show visualization
    )
    
    print(f"TTA individual predictions: {len(tta_pred_individual[0]['ellipse_params'])} total ellipses")
    
    # TTA Prediction with consensuation
    print("\n=== TTA Prediction (Consensuated) ===")
    tta_pred_consensuated = tta_predict(
        model=model,
        image_tensor=img_tensor,
        device=device, 
        min_score=0.75,
        consensuate=True,   # Apply consensuation
        visualize=True      # Show visualization
    )
    
    print(f"TTA consensuated predictions: {len(tta_pred_consensuated[0]['ellipse_params'])} ellipses after consensuation")
    
    # Show final comparison
    print("\n=== Comparison ===")
    print(f"Standard: {len(standard_pred[0]['ellipse_params'])} ellipses")
    print(f"TTA Individual: {len(tta_pred_individual[0]['ellipse_params'])} ellipses") 
    print(f"TTA Consensuated: {len(tta_pred_consensuated[0]['ellipse_params'])} ellipses")
    
    # Plot final consensuated result separately for easy comparison
    if tta_pred_consensuated[0]['ellipse_params'].numel() > 0:
        print("\nShowing final consensuated result:")
        plot_single_pred(img_tensor, tta_pred_consensuated)


# %%
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj=r"C:\Users\mjgto\Desktop\Universidad\5Quinto\TFG_Infor\Proyecto\ellipse-rcnn-FDDB\model.safetensors",
    path_in_repo="model.safetensors",
    repo_id="MJGT/ellipse-rcnn-FDDB",
)
# %%
