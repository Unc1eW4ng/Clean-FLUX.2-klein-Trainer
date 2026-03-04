# Clean FLUX.2-klein Trainer

Most existing frameworks are over-engineered with deep abstractions that make customization a nightmare. This repository provides a clean, flat, and transparent implementation of FLUX.2-klein Training scripts for researchers and creators who want full control over their training logic.

### ✨ Key Features
---
* **Multi-Reference Support**: Native support for **Multi-Reference to One Image** training—perfect for capturing complex features from multiple reference images. It take only 40GB GPU memory to train a 512x512 model with 2 reference images with FLUX.2-klein-4B.
* **Lightweight**: Minimal dependencies. No bloated wrappers, just pure PyTorch and Diffusers logic.

### 🛠️ Quick Start
--- 

**Install Dependencies**

Install torch, torchvision suitable for your CUDA version from https://pytorch.org/get-started/previous-versions/ first.

```bash
# install the latest diffusers main branch (includes Flux2KleinPipeline)
pip install -U "git+https://github.com/huggingface/diffusers.git"

pip install bitsandbytes accelerate transformers diffusers wandb peft
```

**Customize your Training dataset**

take  `data_module.py` as an example (which uses MUSAR, a 2to1 dataset as an example), you can customize your dataset by modifying the `__getitem__` method. The current implementation supports loading multiple reference images and a target image for each training sample.

**Running Training**

```bash
bash train_model.sh
```

**Resume Training**

To resume training from a checkpoint, uncomment the following line in `train_model.sh`:
```
# --resume_from_checkpoint="latest"  \
```

remember to change `$OUTPUT_DIR` to the father directory where your checkpoints are saved like `runs/timestamp`.

### 💡 Model Inference
--- 
After training, you can use the model for inference(included in `inference_klein.py`)

```
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image


pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("pytorch_lora_weights.safetensors")
pipe.to("cuda")
input_image1 = load_image("../ref1.jpg")
input_image2 = load_image("../ref2.jpg")

image = pipe(
  image=[input_image1,input_image2],
  prompt="Your Prompt",
  guidance_scale=1,
  width = 512,
  height = 512,
).images[0]

image.save("output.png")

```

## 🙏 Acknowledgements

This project is built upon the work of the following open-source repositories:
* **[Diffusers](https://github.com/huggingface/diffusers)**:
* **[FLUX.1-Kontext-dev-Training](https://github.com/Bilal143260/FLUX.1-Kontext-dev-Training)**

