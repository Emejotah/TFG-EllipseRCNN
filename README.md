<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>

# Ellipse R-CNN

</div>

A PyTorch (Lightning) implementation of Ellipse R-CNN. Extracted from [another project](https://github.com/wdoppenberg/crater-detection).
The methodology is based on [Ellipse R-CNN: Learning to Infer Elliptical Object from Clustering and Occlusion](https://arxiv.org/abs/2001.11584), albeit
with slight changes. Primarily this implementation was made to enable Crater detection from Moon orbiter sensors, but
works with the [Face Detection Dataset & Benchmark](https://vis-www.cs.umass.edu/fddb/) (FDDB) dataset as well.

<div align="center">

![Sample FDDB predictions](docs/fddb_sample.png)
![Sample crater predictions](docs/craters_sample.png)

</div>

## Installation

```shell
pip install ellipse-rcnn
```

### Optional extras

Enable a feature with the `ellipse-rcnn[<FEATURE>, ...]` pattern.

* `train`: Installs all dependencies necessary to train this model.
* `hf`: Installs `huggingface-hub` and `safetensors` for easy weights saving & loading through the Huggingface platform.


## Quickstart

* Install with all extras through `pip install "ellipse-rcnn[hf,train]"`
* Select the weights you need from my [Huggingface profile](https://huggingface.co/wdoppenberg).

Run the following:

```python
import torch
from ellipse_rcnn.hf import EllipseRCNN  # This is the model with HF functionality included through PyTorchModelHubMixin
from PIL import Image
from torchvision.transforms.functional import to_tensor
from ellipse_rcnn.utils.viz import plot_single_pred


model = EllipseRCNN.from_pretrained("wdoppenberg/crater-rcnn")  # For the crater detection model
model.eval()

png = Image.open("docs/example_craters.png").convert("L")
img = to_tensor(png)
with torch.no_grad():
    pred = model([img])

plot_single_pred(img, pred)
```

This should output the following:

<div align="center">

<img alt="Crater Prediction" height="300" src="docs/crater_pred.png" width="300"/>

</div>


## Training Setup

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. This library's dependencies have mostly
been made optional to make it easy to import the base `EllipseRCNN` model. For training,
more dependencies, most importantly `pytorch-lightning`, are under the `train` group.

For a training setup, run the following from the project root:

```shell
uv sync --extra train
```

This sets up a new virtual environment and installs all packages into it.

Get info on either the training or sample (prediction) script using:

```shell
uv run scripts/train.py --help
# or
uv run scripts/sample.py --help
```

If you want to do experiment tracking, use tensorboard:

```shell
uvx tensorboard --logdir=lightning_logs
```

# Data

Currently the training script only supports training with FDDB. See the required steps for
getting & structuring the data below. More datasets can be supported if needed.
If you want to download a dataset directly, use the following script:

```shell
uv run scripts/download.py <DATASET_NAME> [Optional: --root <ROOT_FOLDER>]
```

## FDDB Data

The [Face Detection Dataset & Benchmark](https://vis-www.cs.umass.edu/fddb/) (FDDB) [module](ellipse_rcnn/data/fddb.py) contains the `Dataset` subclass that does all the data loading and
transformations. It will download and unpack the necessary files to `./data/FDDB`. Simply running the training
script will download the necessary files.

## Craters Data

Unfortunately, the dataset used to create a Crater Detector is not releasable.
To generate it, check out [this module](https://github.com/wdoppenberg/crater-detection/blob/main/src/common/data.py), and ensure
you have a license to run [SurRender](https://www.airbus.com/en/products-services/space/space-customer-support/surrendersoftware).

# Citations

If you use this code in your work, please consider citing the [original paper](https://arxiv.org/abs/2001.11584) and this repository. For this repository,
you can use the following BibTex entry:


```
@misc{EllipseRCNNPyTorch2025,
    author = {Doppenberg, Wouter},
    title = {Ellipse {R-CNN}: {PyTorch} {Implementation}},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/wdoppenberg/ellipse_rcnn}},
    note = {An implementation of the Ellipse R-CNN object detection model in PyTorch, based on "Ellipse R-CNN: Learning to Infer Elliptical Object from Clustering and Occlusion" by Dong et al.}
}
```
