# hf-hub-lightning

<a href="https://colab.research.google.com/github/nateraw/hf-hub-lightning/blob/main/examples/hf_hub_lightning_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A callback for pushing lightning models to the Hugging Face Hub.

**Note:** I made this package for myself, mostly...if folks seem to be interested in it, we'll move this into `huggingface_hub` or an official PyTorch Lightning repo directly and archive this one. Since there are a lot of options/considerations, I decided to just post it separately for now as a proof of concept. 

## Setup

```
pip install hf-hub-lightning
```

## Usage

To periodically upload your model ckpt and TensorBoard logs while training, you can do the following...

```python
import pytorch_lightning as pl
from hf_hub_lightning import HuggingFaceHubCallback

train_dataloader = ...
model = YourCoolModel()
trainer = pl.Trainer(callbacks=[HuggingFaceHubCallback('your_username/model_id')])
trainer.fit(model, train_dataloader)
```

To load your model back from the huggingface hub, just do...

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download('your_username/model_id', 'lit_model.ckpt')
model = YourCoolModel.load_from_checkpoint(ckpt_path)
```
