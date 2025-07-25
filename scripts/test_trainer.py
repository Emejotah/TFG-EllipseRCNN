import torch
from pytorch_lightning import Trainer, LightningModule

class DummyModel(LightningModule):
    def forward(self, x):
        return x
    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

print("Torch sees CUDA:", torch.cuda.is_available())

trainer = Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=1
)

model = DummyModel()
trainer.fit(model, train_dataloaders=[torch.randn(1, 1)])

#%%
import torch
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
else:
    print("No se detectó GPU")

# %%
