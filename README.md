# SimpleSana

## Install
```
git clone git@github.com:AndranikSargsyan/SimpleSana.git
pip install -e .
```

## Usage

```python
import torch
from sana.configs import SANA1_5_1600M_Config, SANA1_5_4800M_Config, SANA1_600M_Config
from sana.pipeline import SanaPipeline

# Use Sana 1.5 1.6B model
sana = SanaPipeline(SANA1_5_1600M_Config())
sana.from_pretrained("hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth")

# Use Sana 1.5 4.8B model
# sana = SanaPipeline(SANA1_5_4800M_Config())
# sana.from_pretrained("hf://Efficient-Large-Model/SANA1.5_4.8B_1024px/checkpoints/SANA1.5_4.8B_1024px.pth")

# Use Sana 1 0.6B model
# sana = SanaPipeline(SANA1_600M_Config())
# sana.from_pretrained("hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth")

image = sana(
    prompt='A small dragon wearing a backpack.',
    height=1024,
    width=1024,
    guidance_scale=3,
    num_inference_steps=14,
    generator=torch.Generator(device='cuda').manual_seed(5),
)

image.save('test.png')
```

